# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the canonical BiRRT planner."""

# Standard Library
import math
from collections.abc import Iterator
from types import SimpleNamespace

# Third Party
import pytest
import torch

# CuRobo
import curobo._src.runtime as curobo_runtime
from curobo._src.cost.criteria_tool_relative_transform import ToolRelativeTransformCriteria
from curobo._src.cost.tool_pose_criteria import PoseErrorFrame, ToolPoseCriteria
from curobo._src.graph_planner.config_graph_planner_resources import GraphPlannerResourcesCfg
from curobo._src.graph_planner.rrt import (
    BiRRTConfig,
    BiRRTPlanner,
    RRTPlannerResources,
)
from curobo._src.graph_planner.rrt.config_rrt_graph_planner import RRTGraphPlannerCfg
from curobo._src.graph_planner.rrt.rrt_projection_sample import (
    RRTSampleProjectionConfig,
    RRTSampleProjectionResult,
)
from curobo._src.state.state_joint import JointState
from curobo._src.trajectory.trajectory_interpolation import TrajInterpolationType
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.types.target_tool_relative_transform import ToolRelativeTransformTarget
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.config_io import load_yaml
from curobo.content import get_robot_configs_path


@pytest.fixture(scope="module")
def resources() -> RRTPlannerResources:
    """Create the shared Franka collision resources."""
    return RRTPlannerResources(
        GraphPlannerResourcesCfg.create(robot="franka.yml", scene_model="collision_test.yml")
    )


@pytest.fixture(scope="module")
def two_environment_resources(resources: RRTPlannerResources) -> RRTPlannerResources:
    """Create one blocked world and one empty world for environment routing."""
    robot_data = load_yaml(str(get_robot_configs_path() / "franka.yml"))
    robot = RobotCfg.create(robot_data, resources.device_cfg, num_envs=2)
    blocked = {
        "cuboid": {
            "blocking_world": {
                "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                "dims": [10.0, 10.0, 10.0],
            }
        }
    }
    return RRTPlannerResources(
        GraphPlannerResourcesCfg.create(
            robot=robot,
            scene_model=[blocked, {}],
            device_cfg=resources.device_cfg,
        )
    )


@pytest.fixture(scope="module")
def shared_goal_routing_resources(
    resources: RRTPlannerResources,
) -> Iterator[RRTPlannerResources]:
    """Create isolated shared-world resources for goal-route integration tests."""
    routing_resources = RRTPlannerResources(
        RRTGraphPlannerCfg.create(
            robot="franka.yml",
            scene_model={},
            device_cfg=resources.device_cfg,
        ).resources_config
    )
    yield routing_resources
    routing_resources.reset_cuda_graph()


@pytest.fixture(scope="module")
def multi_env_goal_routing_resources(
    resources: RRTPlannerResources,
) -> Iterator[RRTPlannerResources]:
    """Create isolated empty per-query worlds for combined route tests."""
    robot_data = load_yaml(str(get_robot_configs_path() / "franka.yml"))
    robot = RobotCfg.create(robot_data, resources.device_cfg, num_envs=2)
    routing_resources = RRTPlannerResources(
        RRTGraphPlannerCfg.create(
            robot=robot,
            scene_model=[{}, {}],
            device_cfg=resources.device_cfg,
        ).resources_config
    )
    yield routing_resources
    routing_resources.reset_cuda_graph()


def _direct_blocked(
    planner: RRTPlannerResources, starts: torch.Tensor, goals: torch.Tensor
) -> torch.Tensor:
    return ~planner.linear_connector.connect(starts, goals).reached


def _path_edges_valid(planner: RRTPlannerResources, path: torch.Tensor) -> bool:
    return not bool(_direct_blocked(planner, path[:-1], path[1:]).any().item())


def _snapshot_batched_tree_storage(workspace) -> tuple[torch.Tensor, ...]:
    """Clone all persistent state owned by both packed trees."""
    return tuple(
        tensor.clone()
        for tree in (workspace.start_tree, workspace.goal_tree)
        for tensor in (tree.nodes, tree.parents, tree.counts)
    )


def _assert_batched_tree_storage(workspace, expected: tuple[torch.Tensor, ...]) -> None:
    """Require byte-for-byte logical tree restoration after executor warmup."""
    actual = tuple(
        tensor
        for tree in (workspace.start_tree, workspace.goal_tree)
        for tensor in (tree.nodes, tree.parents, tree.counts)
    )
    assert len(actual) == len(expected)
    assert all(torch.equal(value, reference) for value, reference in zip(actual, expected))


def _make_fake_batched_workspace(
    resources: RRTPlannerResources,
    batch_size: int,
    sample_count: int,
    executed_samples: list[torch.Tensor],
):
    """Create the minimal packed workspace needed to audit growth inputs."""
    device = resources.device_cfg.device
    dtype = resources.device_cfg.dtype
    dof = resources.action_dim
    capacity = 64

    def make_tree():
        return SimpleNamespace(
            nodes=torch.zeros((batch_size, capacity, dof), device=device, dtype=dtype),
            parents=torch.full((batch_size, capacity), -1, device=device, dtype=torch.int64),
            counts=torch.ones(batch_size, device=device, dtype=torch.int32),
        )

    start_tree = make_tree()
    goal_tree = make_tree()

    class FakeGrowthStep:
        is_initialized = True

        def execute(self, samples, _active, _env_query_idx, _goal_query_idx):
            executed_samples.append(samples.clone())
            start_counts = start_tree.counts.cpu().tolist()
            goal_counts = goal_tree.counts.cpu().tolist()
            return [
                [sample_count, 0, 0, int(start_counts[i]), int(goal_counts[i])]
                for i in range(batch_size)
            ]

    path_reconstruction = SimpleNamespace(
        connection_indices=torch.full((batch_size, 2), -1, device=device, dtype=torch.int64),
        output_paths=torch.empty((batch_size, capacity, dof), device=device, dtype=dtype),
        path_counts=torch.zeros(batch_size, device=device, dtype=torch.int32),
        path_lengths=torch.full((batch_size,), torch.inf, device=device, dtype=dtype),
    )
    return SimpleNamespace(
        batch_size=batch_size,
        start_tree=start_tree,
        goal_tree=goal_tree,
        growth_steps=[FakeGrowthStep(), FakeGrowthStep()],
        query_active=torch.zeros(batch_size, device=device, dtype=torch.bool),
        env_query_idx=torch.zeros(batch_size, device=device, dtype=torch.int32),
        goal_query_idx=torch.zeros(batch_size, device=device, dtype=torch.int32),
        path_reconstruction=path_reconstruction,
        path_counts_host=torch.zeros(batch_size, dtype=torch.int32, pin_memory=True),
        shortcut_status_host=torch.zeros((batch_size, 5), dtype=torch.int32, pin_memory=True),
        direct_status=torch.empty(batch_size, device=device, dtype=torch.int32),
        direct_status_host=torch.empty(batch_size, dtype=torch.int32, pin_memory=True),
        request_query_mask=torch.ones(batch_size, device=device, dtype=torch.bool),
        request_query_mask_all_active=True,
        preprocessing_executors={},
    )


def _make_projection_result(
    original: torch.Tensor,
    projected: torch.Tensor,
    initially_feasible: torch.Tensor,
    feasible: torch.Tensor,
    selected: torch.Tensor,
) -> RRTSampleProjectionResult:
    """Build deterministic projector output for RRT orchestration tests."""
    zeros = torch.zeros(original.shape[0], device=original.device, dtype=original.dtype)
    return RRTSampleProjectionResult(
        samples=projected,
        feasible=feasible,
        initially_feasible=initially_feasible,
        initial_violation=zeros.clone(),
        final_violation=zeros.clone(),
        violation_measured=selected,
        displacement=torch.linalg.vector_norm(projected - original, dim=-1),
    )


def _patch_fake_rrt_search(
    planner: BiRRTPlanner,
    resources: RRTPlannerResources,
    monkeypatch: pytest.MonkeyPatch,
    samples: torch.Tensor,
    executed_samples: list[torch.Tensor],
    *,
    direct_success: bool = False,
):
    """Patch collision-independent direct seeding, sampling, and packed growth steps."""
    monkeypatch.setattr(curobo_runtime, "cuda_graphs", False)
    batch_size, sample_count, _ = samples.shape
    workspace = _make_fake_batched_workspace(
        resources,
        batch_size,
        sample_count,
        executed_samples,
    )
    monkeypatch.setattr(planner, "_get_batched_workspace", lambda _batch_size: workspace)
    monkeypatch.setattr(
        planner,
        "_generate_proposal_batch",
        lambda *_args: samples.clone(),
    )
    monkeypatch.setattr(
        resources,
        "check_samples_feasibility",
        lambda actions, **_kwargs: torch.ones(
            actions.shape[0], device=actions.device, dtype=torch.bool
        ),
    )

    def fake_direct_edges(starts, desired, **_kwargs):
        reached = torch.full(
            (2 * starts.shape[0],), direct_success, device=starts.device, dtype=torch.bool
        )
        progressed = reached.clone()
        endpoints = (
            torch.cat((desired, starts), dim=0)
            if direct_success
            else torch.cat((starts, desired), dim=0)
        )
        return endpoints, reached, progressed, torch.ones_like(reached)

    def fake_reconstruct(fake_workspace, records):
        for record in records:
            record.path = torch.stack(
                (
                    fake_workspace.start_tree.nodes[record.lane, 0],
                    fake_workspace.goal_tree.nodes[record.lane, 0],
                )
            )
            fake_workspace.path_reconstruction.path_lengths[record.lane] = (
                torch.linalg.vector_norm(record.path[1] - record.path[0])
            )
            fake_workspace.path_reconstruction.path_counts[record.lane] = record.path.shape[0]

    monkeypatch.setattr(
        planner,
        "_evaluate_staged_direct",
        lambda connector: fake_direct_edges(
            connector.request_starts,
            connector.request_goals,
        ),
    )
    monkeypatch.setattr(planner, "_reconstruct_growth_paths", fake_reconstruct)
    return workspace


def test_default_fits_one_feasibility_buffer(resources: RRTPlannerResources):
    """Use the canonical 64-state, 0.02-rad, 32-step profile."""
    config = BiRRTConfig()

    assert config.samples_per_iteration == 24
    assert config.cspace_resolution == pytest.approx(0.02)
    assert config.fixed_steer_steps == 64
    assert config.max_extension_distance == pytest.approx((64 - 2) * 0.02)
    assert config.failure_extension_iterations == 20
    assert config.max_iterations + config.failure_extension_iterations == 32
    assert config.max_nodes_per_tree == 1024
    assert config.max_nodes_per_tree >= 2 + config.samples_per_iteration * 32
    assert config.shortcut_path
    assert config.shortcut_path_capacity == 16
    assert config.sample_projection is None
    assert (
        config.samples_per_iteration * config.fixed_steer_steps
        <= resources.config.feasibility_buffer_size
    )


def test_growth_outputs_alias_append_and_passive_input_buffers(
    resources: RRTPlannerResources,
) -> None:
    """Write steering results directly into graph-stable tree append storage."""
    planner = BiRRTPlanner(resources)
    workspace = planner._get_batched_workspace(2)

    for growth_step in workspace.growth_steps:
        active = growth_step._stage_buffers["active"]
        passive = growth_step._stage_buffers["passive"]
        assert active["endpoints"].data_ptr() == growth_step.active_append.candidates.data_ptr()
        assert active["endpoints"].data_ptr() == growth_step.passive_nearest.desired.data_ptr()
        assert (
            active["progressed"].data_ptr() == growth_step.active_append.candidate_valid.data_ptr()
        )
        assert passive["endpoints"].data_ptr() == growth_step.passive_append.candidates.data_ptr()
        assert passive["progressed"].data_ptr() == (
            growth_step.passive_append.candidate_valid.data_ptr()
        )
        assert growth_step.active_nearest.parent_indices.data_ptr() == (
            growth_step.active_append.candidate_parents.data_ptr()
        )
        assert growth_step.passive_nearest.parent_indices.data_ptr() == (
            growth_step.passive_append.candidate_parents.data_ptr()
        )


@pytest.mark.parametrize("projected", (False, True))
def test_rrt_growth_reuses_cached_bounded_uniform_samples(
    resources: RRTPlannerResources,
    projected: bool,
) -> None:
    """Use the same zero-copy bounded Halton stream for standard and projected RRT."""
    projection = RRTSampleProjectionConfig() if projected else None
    planner = BiRRTPlanner(
        resources,
        BiRRTConfig(shortcut_path=False, sample_projection=projection),
    )
    sample_buffer = resources.uniform_sampler.action_sample_generator

    assert sample_buffer._bounded_sample_buffer is not None
    planner.reset_seed()
    samples = planner._sample_batched(batch_size=1)

    assert samples.data_ptr() == sample_buffer._bounded_sample_buffer.data_ptr()


def test_rrt_config_installs_disabled_tool_pose_constraint():
    config = RRTGraphPlannerCfg.create(robot="franka.yml")

    tool_pose = config.resources_config.rollout_config.constraint_cfg.tool_pose_cfg

    assert tool_pose is not None
    assert tool_pose.weight.tolist() == [1.0, 1.0]
    assert (
        torch.count_nonzero(tool_pose.default_tool_pose_criteria.terminal_position_projector) == 0
    )
    assert (
        torch.count_nonzero(tool_pose.default_tool_pose_criteria.terminal_rotation_projector) == 0
    )
    assert torch.count_nonzero(tool_pose.default_tool_pose_criteria.path_position_projector) == 0
    assert torch.count_nonzero(tool_pose.default_tool_pose_criteria.path_rotation_projector) == 0
    assert config.resources_config.rollout_config.constraint_cfg.scene_collision_cfg is not None
    assert not (
        config.resources_config.rollout_config.constraint_cfg.scene_collision_cfg.convert_to_binary
    )
    assert config.resources_config.rollout_config.constraint_cfg.self_collision_cfg is not None
    assert not (
        config.resources_config.rollout_config.constraint_cfg.self_collision_cfg.convert_to_binary
    )


def _assert_tool_pose_goal_routes(
    resources: RRTPlannerResources,
    env_query_idx: torch.Tensor | None,
) -> None:
    resources.reset_seed()
    configurations = resources.uniform_sampler.generate_feasible_action_samples(32)[[0, -1]]
    joint_state = JointState.from_position(
        configurations,
        joint_names=resources.kinematics.joint_names,
    )
    goal_tool_poses = resources.kinematics.compute_kinematics(joint_state).tool_poses.as_goal()
    identity = torch.eye(3, **resources.device_cfg.as_torch_dict())
    criteria = {
        frame: ToolPoseCriteria(
            path_position_projector=identity,
            path_rotation_projector=identity,
            path_tolerance=[0.001, 0.001],
            device_cfg=resources.device_cfg,
        )
        for frame in resources.kinematics.tool_frames
    }
    planner = BiRRTPlanner(resources)
    planner.update_tool_pose_constraint(goal_tool_poses, criteria)

    correct_goal_routes = torch.tensor(
        [0, 1],
        device=configurations.device,
        dtype=torch.int32,
    )
    wrong_goal_routes = torch.zeros_like(correct_goal_routes)
    correct = resources.check_samples_feasibility(
        configurations,
        env_query_idx=env_query_idx,
        goal_query_idx=correct_goal_routes,
    )
    wrong = resources.check_samples_feasibility(
        configurations,
        env_query_idx=env_query_idx,
        goal_query_idx=wrong_goal_routes,
    )

    assert resources.feasibility_rollout.has_goal_pose_routes
    assert correct.tolist() == [True, True]
    assert wrong.tolist() == [True, False]


def test_tool_pose_goal_routes_in_shared_environment(
    shared_goal_routing_resources: RRTPlannerResources,
) -> None:
    """Route each constrained query to its own target in one shared world."""
    _assert_tool_pose_goal_routes(shared_goal_routing_resources, env_query_idx=None)


def test_tool_pose_goal_routes_in_multiple_environments(
    multi_env_goal_routing_resources: RRTPlannerResources,
) -> None:
    """Keep goal and environment routes independent on the combined executor."""
    env_query_idx = torch.tensor(
        [0, 1],
        device=multi_env_goal_routing_resources.device_cfg.device,
        dtype=torch.int32,
    )
    _assert_tool_pose_goal_routes(multi_env_goal_routing_resources, env_query_idx)


@pytest.mark.parametrize("enable_dynamics", [False, True])
def test_inertial_updates_are_routed_to_every_rrt_rollout(enable_dynamics: bool) -> None:
    """Update enabled dynamics models and skip kinematics-only RRT transitions."""

    class CapturingTransition:
        def __init__(self) -> None:
            self.compute_inverse_dynamics = enable_dynamics
            self.single_calls = []
            self.batch_calls = []

        def update_link_inertial(self, link_name, mass, com, inertia) -> None:
            self.single_calls.append((link_name, mass, com, inertia))

        def update_links_inertial(self, link_properties) -> None:
            self.batch_calls.append(link_properties)

    rollouts = [
        SimpleNamespace(
            transition_model=CapturingTransition(),
            metrics_transition_model=CapturingTransition(),
        )
        for _ in range(5)
    ]
    planner = object.__new__(BiRRTPlanner)
    planner.resources = SimpleNamespace(get_all_rollout_instances=lambda: rollouts[:2])
    planner._direct_connectors = {1: SimpleNamespace(rollout=rollouts[4])}
    planner._workspaces = {
        4: SimpleNamespace(
            growth_steps=[SimpleNamespace(rollout=rollouts[2])],
            finalization_rollout=rollouts[3],
        )
    }

    planner.update_link_inertial("attached_object", 3.0)
    properties = {"attached_object": {"mass": 2.0}}
    planner.update_links_inertial(properties)

    for rollout in rollouts:
        for model in (rollout.transition_model, rollout.metrics_transition_model):
            assert model.single_calls == (
                [("attached_object", 3.0, None, None)] if enable_dynamics else []
            )
            assert model.batch_calls == ([properties] if enable_dynamics else [])


def test_tool_pose_constraint_is_routed_to_every_rrt_rollout_and_can_be_disabled():
    """Keep projection and edge rollouts on one runtime task-manifold definition."""

    class CapturingRollout:
        def __init__(self):
            self.goals = []
            self.criteria = []
            self.aliased_goals = []

        def update_params(self, goal):
            self.goals.append(goal)

        def update_params_cost_managers(self, *, tool_pose_criteria):
            self.criteria.append(tool_pose_criteria)

        def alias_goal_tool_poses(self, goal_tool_poses):
            self.aliased_goals.append(goal_tool_poses)

    feasibility_rollout = CapturingRollout()
    auxiliary_rollout = CapturingRollout()
    growth_rollout = CapturingRollout()
    planner = object.__new__(BiRRTPlanner)
    device_cfg = DeviceCfg(device=torch.device("cpu"))
    planner.resources = SimpleNamespace(
        device_cfg=device_cfg,
        kinematics=SimpleNamespace(tool_frames=["ee_link"]),
        config=SimpleNamespace(
            feasibility_buffer_size=17,
            rollout_config=SimpleNamespace(
                constraint_cfg=SimpleNamespace(tool_pose_cfg=SimpleNamespace())
            ),
        ),
        feasibility_rollout=feasibility_rollout,
        auxiliary_rollout=auxiliary_rollout,
    )
    planner._constraint_goal_tool_poses = None
    planner._constraint_tool_pose_criteria = None
    planner._constraint_tool_relative_transform_target = None
    planner._constraint_tool_relative_transform_criteria = None
    planner._sample_projector = SimpleNamespace(
        config=SimpleNamespace(
            projection_slots=3,
            line_search_scales=(0.0, 1.0),
        )
    )
    planner._workspaces = {
        2: SimpleNamespace(
            growth_steps=[
                SimpleNamespace(
                    rollout=growth_rollout,
                    batch_size=2,
                    sample_count=4,
                    max_steps=5,
                )
            ]
        )
    }
    reset_calls = []
    planner.reset_cuda_graph = lambda: reset_calls.append(True)
    goal = GoalToolPose(
        tool_frames=["ee_link"],
        position=torch.zeros((1, 1, 1, 1, 3)),
        quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0]).reshape(1, 1, 1, 1, 4),
    )
    criteria = ToolPoseCriteria(
        path_position_projector=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        path_position_axis_scale=[1.0, 0.5, 1.0],
        path_rotation_projector=torch.eye(3),
        path_rotation_axis_scale=[0.4, 0.3, 0.2],
        terminal_tolerance=[0.001, 0.002],
        path_tolerance=[0.01, 0.02],
        error_frame=PoseErrorFrame.GOAL,
        device_cfg=device_cfg,
    )

    planner.update_tool_pose_constraint(goal, {"ee_link": criteria})

    assert reset_calls == [True]
    assert planner._constraint_goal_tool_poses is not goal
    assert planner._constraint_tool_pose_criteria["ee_link"] is not criteria
    stored_goal_position_ptr = planner._constraint_goal_tool_poses.position.data_ptr()
    stored_criteria_ptrs = {
        name: getattr(planner._constraint_tool_pose_criteria["ee_link"], name).data_ptr()
        for name in ToolPoseCriteria.tensor_field_names()
    }
    expected_batches = (
        (feasibility_rollout, 17),
        (auxiliary_rollout, 3),
        (growth_rollout, 40),
    )
    for rollout, batch_size in expected_batches:
        assert len(rollout.goals) == 1
        routed_goal = rollout.goals[0]
        assert routed_goal.link_goal_poses is planner._constraint_goal_tool_poses
        assert routed_goal.idxs_link_pose.shape == (batch_size, 1)
        assert torch.count_nonzero(routed_goal.idxs_link_pose) == 0
        assert rollout.aliased_goals == [planner._constraint_goal_tool_poses]
        assert len(rollout.criteria) == 1
        routed_criteria = rollout.criteria[0]["ee_link"]
        for terminal_name, path_name in (
            ("terminal_position_projector", "path_position_projector"),
            ("terminal_rotation_projector", "path_rotation_projector"),
            ("terminal_position_axis_scale", "path_position_axis_scale"),
            ("terminal_rotation_axis_scale", "path_rotation_axis_scale"),
            ("terminal_tolerance", "path_tolerance"),
        ):
            torch.testing.assert_close(
                getattr(routed_criteria, terminal_name), getattr(criteria, path_name)
            )
            torch.testing.assert_close(
                getattr(routed_criteria, path_name), getattr(criteria, path_name)
            )
        assert routed_criteria.error_frame.item() == PoseErrorFrame.GOAL

    updated_goal = goal.clone()
    updated_goal.position.fill_(1.0)
    planner.update_tool_pose_constraint_goal(updated_goal)

    assert reset_calls == [True]
    assert planner._constraint_goal_tool_poses.position.data_ptr() == stored_goal_position_ptr
    torch.testing.assert_close(
        planner._constraint_goal_tool_poses.position,
        updated_goal.position,
    )
    for rollout, batch_size in expected_batches:
        assert len(rollout.goals) == 1
        routed_goal = rollout.goals[0]
        assert routed_goal.link_goal_poses is planner._constraint_goal_tool_poses
        assert routed_goal.idxs_link_pose.shape == (batch_size, 1)
        assert len(rollout.criteria) == 1

    updated_criteria = ToolPoseCriteria.linear_motion(
        direction=[1.0, 2.0, 3.0],
        rotation_axis=[-3.0, 1.0, 2.0],
        error_frame=PoseErrorFrame.WORLD,
        device_cfg=device_cfg,
    )
    planner.update_tool_pose_constraint_criteria({"ee_link": updated_criteria})

    assert {
        name: getattr(planner._constraint_tool_pose_criteria["ee_link"], name).data_ptr()
        for name in ToolPoseCriteria.tensor_field_names()
    } == stored_criteria_ptrs
    for rollout, _ in expected_batches:
        assert len(rollout.goals) == 1
        assert len(rollout.criteria) == 2
        routed_criteria = rollout.criteria[-1]["ee_link"]
        torch.testing.assert_close(
            routed_criteria.terminal_position_projector,
            updated_criteria.path_position_projector,
        )
        torch.testing.assert_close(
            routed_criteria.terminal_rotation_projector,
            updated_criteria.path_rotation_projector,
        )
        assert routed_criteria.error_frame.item() == PoseErrorFrame.WORLD

    replacement_goal = updated_goal.clone()
    replacement_goal.position.fill_(2.0)
    planner.update_tool_pose_constraint(replacement_goal, {"ee_link": criteria})

    assert planner._constraint_goal_tool_poses.position.data_ptr() == stored_goal_position_ptr
    torch.testing.assert_close(
        planner._constraint_goal_tool_poses.position,
        replacement_goal.position,
    )
    for rollout, _ in expected_batches:
        assert len(rollout.goals) == 1
        assert len(rollout.criteria) == 3

    planner.disable_tool_pose_constraint()

    for rollout, _ in expected_batches:
        assert len(rollout.criteria) == 4
        disabled = rollout.criteria[-1]["ee_link"]
        assert torch.count_nonzero(disabled.terminal_position_projector) == 0
        assert torch.count_nonzero(disabled.terminal_rotation_projector) == 0
        assert torch.count_nonzero(disabled.path_position_projector) == 0
        assert torch.count_nonzero(disabled.path_rotation_projector) == 0


def test_relative_transform_constraint_is_routed_to_every_rrt_rollout_and_can_be_disabled():
    class CapturingRollout:
        def __init__(self):
            self.goals = []
            self.criteria = []

        def update_params(self, goal):
            self.goals.append(goal)

        def update_params_cost_managers(self, **criteria):
            self.criteria.append(criteria)

    feasibility_rollout = CapturingRollout()
    auxiliary_rollout = CapturingRollout()
    growth_rollout = CapturingRollout()
    planner = object.__new__(BiRRTPlanner)
    device_cfg = DeviceCfg(device=torch.device("cpu"))
    relative_cfg = SimpleNamespace(reference_tool_frame="reference")
    planner.resources = SimpleNamespace(
        device_cfg=device_cfg,
        kinematics=SimpleNamespace(tool_frames=["reference", "tool"]),
        config=SimpleNamespace(
            feasibility_buffer_size=11,
            rollout_config=SimpleNamespace(
                constraint_cfg=SimpleNamespace(
                    tool_pose_cfg=None,
                    tool_relative_transform_cfg=relative_cfg,
                )
            ),
        ),
        feasibility_rollout=feasibility_rollout,
        auxiliary_rollout=auxiliary_rollout,
    )
    planner._constraint_goal_tool_poses = None
    planner._constraint_tool_pose_criteria = None
    planner._constraint_tool_relative_transform_target = None
    planner._constraint_tool_relative_transform_criteria = None
    planner._sample_projector = SimpleNamespace(
        config=SimpleNamespace(projection_slots=2, line_search_scales=(0.0, 0.5, 1.0))
    )
    planner._workspaces = {
        1: SimpleNamespace(
            growth_steps=[
                SimpleNamespace(
                    rollout=growth_rollout,
                    batch_size=1,
                    sample_count=3,
                    max_steps=7,
                )
            ]
        )
    }
    reset_calls = []
    planner.reset_cuda_graph = lambda: reset_calls.append(True)
    position = torch.zeros((1, 2, 3))
    position[:, 1, 0] = 0.2
    quaternion = torch.zeros((1, 2, 4))
    quaternion[..., 0] = 1.0
    target = ToolRelativeTransformTarget(
        reference_tool_frame="reference",
        tool_frames=["reference", "tool"],
        position=position,
        quaternion=quaternion,
    )
    criteria = {
        "reference": ToolRelativeTransformCriteria.disabled(device_cfg=device_cfg),
        "tool": ToolRelativeTransformCriteria.rigid(device_cfg=device_cfg),
    }

    planner.update_tool_relative_transform_constraint(target, criteria)

    assert reset_calls == [True]
    assert planner._constraint_tool_relative_transform_target is not target
    expected_batches = (
        (feasibility_rollout, 11),
        (auxiliary_rollout, 4),
        (growth_rollout, 21),
    )
    for rollout, batch_size in expected_batches:
        routed_goal = rollout.goals[-1]
        assert (
            routed_goal.tool_relative_transform_target
            is planner._constraint_tool_relative_transform_target
        )
        assert routed_goal.idxs_tool_relative_transform_target.shape == (batch_size, 1)
        assert torch.count_nonzero(routed_goal.idxs_tool_relative_transform_target) == 0
        routed_criteria = rollout.criteria[-1]["tool_relative_transform_criteria"]
        assert routed_criteria["tool"] is not criteria["tool"]
        assert routed_criteria["tool"].enabled

    planner.disable_tool_relative_transform_constraint()

    for rollout, _ in expected_batches:
        disabled = rollout.criteria[-1]["tool_relative_transform_criteria"]
        assert not disabled["tool"].enabled


def test_combined_absolute_and_relative_rrt_goal_preserves_rollout_batch_size():
    """A shared absolute goal must not collapse batched relative routing."""

    class CapturingRollout:
        def __init__(self):
            self.goal = None
            self.aliased_goal = None

        def update_params(self, goal):
            self.goal = goal

        def alias_goal_tool_poses(self, goal_tool_poses):
            self.aliased_goal = goal_tool_poses

    planner = object.__new__(BiRRTPlanner)
    planner.resources = SimpleNamespace(
        device_cfg=DeviceCfg(device=torch.device("cpu")),
    )
    planner._constraint_goal_tool_poses = GoalToolPose(
        tool_frames=["reference", "tool"],
        position=torch.zeros((1, 1, 2, 1, 3)),
        quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0])
        .reshape(1, 1, 1, 1, 4)
        .repeat(
            1,
            1,
            2,
            1,
            1,
        ),
    )
    planner._constraint_tool_relative_transform_target = ToolRelativeTransformTarget(
        reference_tool_frame="reference",
        tool_frames=["reference", "tool"],
        position=torch.zeros((1, 2, 3)),
        quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0]).reshape(1, 1, 4).repeat(1, 2, 1),
    )
    rollout = CapturingRollout()

    planner._apply_constraint_goal(rollout, batch_size=7)

    assert rollout.goal.batch_size == 7
    assert rollout.goal.idxs_link_pose.shape == (7, 1)
    assert rollout.goal.idxs_tool_relative_transform_target.shape == (7, 1)
    assert rollout.aliased_goal is planner._constraint_goal_tool_poses


def test_projection_disabled_never_calls_projector_and_reports_zero_work(
    resources: RRTPlannerResources,
    monkeypatch: pytest.MonkeyPatch,
):
    """Keep the ordinary RRT path free of projection calls and projector state."""
    sample_count = 4
    config = BiRRTConfig(
        samples_per_iteration=sample_count,
        max_iterations=1,
        failure_extension_iterations=0,
        max_nodes_per_tree=64,
        shortcut_path=False,
    )
    planner = BiRRTPlanner(resources, config)
    device = resources.device_cfg.device
    dtype = resources.device_cfg.dtype
    samples = torch.arange(
        sample_count * resources.action_dim,
        device=device,
        dtype=dtype,
    ).reshape(1, sample_count, resources.action_dim)
    executed_samples: list[torch.Tensor] = []
    _patch_fake_rrt_search(planner, resources, monkeypatch, samples, executed_samples)

    def fail_projection(*_args, **_kwargs):
        raise AssertionError("projection must not run when sample_projection is unset")

    monkeypatch.setattr(planner, "project_samples", fail_projection)
    starts = torch.zeros((1, resources.action_dim), device=device, dtype=dtype)
    goals = torch.ones_like(starts)
    result = planner.find_path_batched(
        starts,
        goals,
        interpolate_waypoints=False,
        validate_interpolated_trajectory=False,
    )

    assert len(executed_samples) == 1
    torch.testing.assert_close(executed_samples[0], samples, rtol=0.0, atol=0.0)
    assert planner._sample_projector is None
    assert result.debug_info["projection"] == {
        "enabled": False,
        "calls": 0,
        "eligible_candidates": 0,
        "initially_invalid": 0,
        "selected": 0,
        "repaired": 0,
        "substituted": 0,
        "displacement_sum": 0.0,
        "displacement_max": 0.0,
    }


def test_projected_growth_excludes_inactive_lanes_from_eligibility(
    resources: RRTPlannerResources,
    monkeypatch: pytest.MonkeyPatch,
):
    """Pass every proposal to one fixed call but make only active lanes eligible."""
    batch_size = 2
    sample_count = 4
    planner = BiRRTPlanner(
        resources,
        BiRRTConfig(
            samples_per_iteration=sample_count,
            max_iterations=1,
            failure_extension_iterations=0,
            max_nodes_per_tree=64,
            shortcut_path=False,
            sample_projection=RRTSampleProjectionConfig(),
        ),
    )
    device = resources.device_cfg.device
    dtype = resources.device_cfg.dtype
    samples = torch.arange(
        batch_size * sample_count * resources.action_dim,
        device=device,
        dtype=dtype,
    ).reshape(batch_size, sample_count, resources.action_dim)
    executed_samples: list[torch.Tensor] = []
    _patch_fake_rrt_search(planner, resources, monkeypatch, samples, executed_samples)
    captured = {}

    def fake_projection(flat_samples, *, env_query_idx=None, eligible_mask=None, **_kwargs):
        captured["samples"] = flat_samples.clone()
        captured["env_query_idx"] = env_query_idx.clone()
        captured["eligible_mask"] = eligible_mask.clone()
        feasible = torch.ones(flat_samples.shape[0], device=device, dtype=torch.bool)
        selected = torch.zeros_like(feasible)
        return _make_projection_result(
            flat_samples,
            flat_samples.clone(),
            feasible,
            feasible.clone(),
            selected,
        )

    monkeypatch.setattr(planner, "project_samples", fake_projection)
    starts = torch.zeros((batch_size, resources.action_dim), device=device, dtype=dtype)
    goals = torch.ones_like(starts)
    result = planner.find_path_batched(
        starts,
        goals,
        query_mask=torch.tensor([True, False], device=device, dtype=torch.bool),
        interpolate_waypoints=False,
        validate_interpolated_trajectory=False,
    )

    torch.testing.assert_close(captured["samples"], samples.reshape(-1, resources.action_dim))
    assert torch.equal(
        captured["eligible_mask"],
        torch.tensor(
            [True] * sample_count + [False] * sample_count,
            device=device,
            dtype=torch.bool,
        ),
    )
    assert torch.equal(
        captured["env_query_idx"],
        torch.zeros(batch_size * sample_count, device=device, dtype=torch.int32),
    )
    assert len(executed_samples) == 1
    torch.testing.assert_close(executed_samples[0], samples, rtol=0.0, atol=0.0)
    assert result.debug_info["projection"] == {
        "enabled": True,
        "calls": 1,
        "eligible_candidates": sample_count,
        "initially_invalid": 0,
        "selected": 0,
        "repaired": 0,
        "substituted": 0,
        "displacement_sum": 0.0,
        "displacement_max": 0.0,
    }


def test_projected_growth_substitutes_only_exact_repairs(
    resources: RRTPlannerResources,
    monkeypatch: pytest.MonkeyPatch,
):
    """Preserve valid and failed candidates even if optimizer output moved them."""
    sample_count = 4
    planner = BiRRTPlanner(
        resources,
        BiRRTConfig(
            samples_per_iteration=sample_count,
            max_iterations=1,
            failure_extension_iterations=0,
            max_nodes_per_tree=64,
            shortcut_path=False,
            sample_projection=RRTSampleProjectionConfig(),
        ),
    )
    device = resources.device_cfg.device
    dtype = resources.device_cfg.dtype
    samples = torch.zeros((1, sample_count, resources.action_dim), device=device, dtype=dtype)
    samples[0, :, 0] = torch.arange(sample_count, device=device, dtype=dtype)
    executed_samples: list[torch.Tensor] = []
    _patch_fake_rrt_search(planner, resources, monkeypatch, samples, executed_samples)

    def fake_projection(flat_samples, *, env_query_idx=None, eligible_mask=None, **_kwargs):
        assert env_query_idx is not None
        assert torch.all(eligible_mask)
        projected = flat_samples.clone()
        projected[:, 0] += torch.tensor([9.0, 1.0, 7.0, 2.0], device=device, dtype=dtype)
        initially_feasible = torch.tensor(
            [True, False, False, False], device=device, dtype=torch.bool
        )
        feasible = torch.tensor([True, True, False, True], device=device, dtype=torch.bool)
        selected = torch.tensor([False, True, True, True], device=device, dtype=torch.bool)
        return _make_projection_result(
            flat_samples,
            projected,
            initially_feasible,
            feasible,
            selected,
        )

    monkeypatch.setattr(planner, "project_samples", fake_projection)
    starts = torch.zeros((1, resources.action_dim), device=device, dtype=dtype)
    goals = torch.ones_like(starts)
    result = planner.find_path_batched(
        starts,
        goals,
        interpolate_waypoints=False,
        validate_interpolated_trajectory=False,
    )

    expected = samples.clone()
    expected[0, 1, 0] += 1.0
    expected[0, 3, 0] += 2.0
    assert len(executed_samples) == 1
    torch.testing.assert_close(executed_samples[0], expected, rtol=0.0, atol=0.0)
    assert result.debug_info["projection"] == {
        "enabled": True,
        "calls": 1,
        "eligible_candidates": sample_count,
        "initially_invalid": 3,
        "selected": 3,
        "repaired": 2,
        "substituted": 2,
        "displacement_sum": pytest.approx(3.0),
        "displacement_max": pytest.approx(2.0),
    }


def test_direct_success_skips_projected_growth_and_reports_zero_projection_calls(
    resources: RRTPlannerResources,
    monkeypatch: pytest.MonkeyPatch,
):
    """Do not pay projection cost when direct seeding already solved the query."""
    sample_count = 4
    planner = BiRRTPlanner(
        resources,
        BiRRTConfig(
            samples_per_iteration=sample_count,
            max_iterations=1,
            failure_extension_iterations=0,
            max_nodes_per_tree=64,
            shortcut_path=False,
            sample_projection=RRTSampleProjectionConfig(),
        ),
    )
    device = resources.device_cfg.device
    dtype = resources.device_cfg.dtype
    samples = torch.zeros((1, sample_count, resources.action_dim), device=device, dtype=dtype)
    executed_samples: list[torch.Tensor] = []
    _patch_fake_rrt_search(
        planner,
        resources,
        monkeypatch,
        samples,
        executed_samples,
        direct_success=True,
    )

    def fail_projection(*_args, **_kwargs):
        raise AssertionError("direct success must not invoke projection")

    monkeypatch.setattr(planner, "project_samples", fail_projection)
    starts = torch.zeros((1, resources.action_dim), device=device, dtype=dtype)
    goals = torch.ones_like(starts)
    result = planner.find_path_batched(
        starts,
        goals,
        interpolate_waypoints=False,
        validate_interpolated_trajectory=False,
    )

    assert result.success.tolist() == [True]
    assert result.debug_info["growth_steps"] == 0
    assert not executed_samples
    assert result.debug_info["projection"] == {
        "enabled": True,
        "calls": 0,
        "eligible_candidates": 0,
        "initially_invalid": 0,
        "selected": 0,
        "repaired": 0,
        "substituted": 0,
        "displacement_sum": 0.0,
        "displacement_max": 0.0,
    }


def test_rrt_steering_profile_derives_safe_extension_and_preserves_override():
    derived = BiRRTConfig(cspace_resolution=0.1, fixed_steer_steps=10)
    explicit = BiRRTConfig(max_extension_distance=0.4)

    assert derived.max_extension_distance == pytest.approx(0.8)
    assert explicit.max_extension_distance == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"samples_per_iteration": 0}, "samples_per_iteration"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"failure_extension_iterations": -1}, "failure_extension_iterations"),
        ({"max_batched_lanes": 3}, "max_batched_lanes"),
        ({"fixed_steer_steps": 2}, "fixed_steer_steps"),
        ({"cspace_resolution": 0.0}, "cspace_resolution"),
        ({"max_extension_distance": 1.25}, "max_extension_distance"),
        ({"shortcut_path_capacity": 2}, "shortcut_path_capacity"),
        ({"shortcut_path_capacity": True}, "shortcut_path_capacity"),
        ({"shortcut_state_capacity": 1}, "shortcut_state_capacity"),
        ({"goal_bias_count": 25}, "goal_bias_count"),
        ({"goal_bias_count": 1.0}, "goal_bias_count"),
        ({"profile_cuda_event_phase": "invalid"}, "profile_cuda_event_phase"),
        ({"profile_cuda_events": 1}, "profile_cuda_events"),
        ({"measure_shortcut_time": 1}, "measure_shortcut_time"),
    ),
)
def test_rrt_rejects_invalid_canonical_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        BiRRTConfig(**kwargs)


def test_rrt_rejects_insufficient_tree_capacity():
    with pytest.raises(ValueError, match="max_nodes_per_tree"):
        BiRRTConfig(
            samples_per_iteration=4,
            max_iterations=2,
            failure_extension_iterations=1,
            max_nodes_per_tree=13,
        )


@pytest.mark.parametrize("max_batch_size", (0, 129, True, 1.5))
def test_rrt_warmup_rejects_invalid_lane_width_before_work(
    resources: RRTPlannerResources,
    max_batch_size,
):
    planner = BiRRTPlanner(resources)

    with pytest.raises(ValueError, match="max_batch_size"):
        planner.warmup(num_warmup_iterations=0, max_batch_size=max_batch_size)


def test_rrt_warmup_fills_samples_rejected_by_feasibility() -> None:
    """Keep warmup fixed-shape when an obstructed scene rejects bounded samples."""

    class WarmupSampler:
        def generate_feasible_action_samples(self, action_count: int) -> torch.Tensor:
            assert action_count == 3
            return torch.tensor([[1.0, 2.0]])

        def generate_action_samples(self, action_count: int, bounded: bool) -> torch.Tensor:
            assert action_count == 2
            assert bounded
            return torch.tensor([[3.0, 4.0], [5.0, 6.0]])

    planner = object.__new__(BiRRTPlanner)
    planner.resources = SimpleNamespace(uniform_sampler=WarmupSampler())

    actions = planner._generate_warmup_actions(3)

    torch.testing.assert_close(
        actions,
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"sampling_policy": "invalid"}, "sampling_policy"),
        ({"informed_sample_count": 0}, "informed_sample_count"),
        ({"informed_sample_count": True}, "informed_sample_count"),
        ({"informed_initial_scale": 0.9}, "informed_initial_scale"),
        ({"informed_growth_factor": float("inf")}, "informed_growth_factor"),
        ({"informed_growth_interval": 0}, "informed_growth_interval"),
        ({"informed_min_slack": -0.1}, "informed_min_slack"),
        (
            {"informed_initial_scale": 2.0, "informed_max_scale": 1.5},
            "informed_max_scale",
        ),
    ),
)
def test_rrt_rejects_invalid_informed_sampling_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        BiRRTConfig(**kwargs)


def test_rrt_rejects_overlapping_mixed_sample_regions():
    with pytest.raises(ValueError, match="goal_bias_count.*informed_sample_count"):
        BiRRTConfig(
            samples_per_iteration=4,
            goal_bias_count=2,
            informed_sample_count=3,
            sampling_policy="mixed_ellipsoid",
        )


def test_linear_connector_resolution_override_changes_only_line_density(
    resources: RRTPlannerResources,
):
    start = torch.zeros(
        (1, resources.action_dim),
        device=resources.device_cfg.device,
        dtype=resources.device_cfg.dtype,
    )
    goal = start.clone()
    goal[:, 0] = 0.08
    weighted_distance = float(
        torch.max(torch.abs((goal - start) * resources.cspace_distance_weight)).item()
    )

    default_line = resources.linear_connector._compute_steering_line_points(start, goal)
    rrt_line = resources.linear_connector._compute_steering_line_points(
        start, goal, cspace_resolution=0.02
    )

    assert (
        default_line.shape[1]
        == math.ceil(weighted_distance / resources.config.cspace_similarity_threshold) + 1
    )
    assert rrt_line.shape[1] == math.ceil(weighted_distance / 0.02) + 1
    assert rrt_line.shape[1] < default_line.shape[1]


def test_mixed_sampler_reset_seed_reproduces_bounded_proposal_transforms(
    resources: RRTPlannerResources,
):
    planner = BiRRTPlanner(
        resources,
        BiRRTConfig(
            sampling_policy="mixed_ellipsoid",
            goal_bias_count=0,
            informed_sample_count=8,
        ),
    )
    batch_size = 3
    lows = resources.action_bound_lows
    highs = resources.action_bound_highs
    action_range = highs - lows
    starts = (
        lows
        + torch.tensor([[0.20], [0.35], [0.45]], device=lows.device, dtype=lows.dtype)
        * action_range
    )
    goals = (
        lows
        + torch.tensor([[0.80], [0.65], [0.55]], device=lows.device, dtype=lows.dtype)
        * action_range
    )
    primary_buffer = resources.uniform_sampler.action_sample_generator
    latent_schedule = planner._informed_latent_schedule
    assert latent_schedule is not None

    planner.reset_seed()
    primary_initial = primary_buffer._int_gen.get_state().clone()
    latent_initial = latent_schedule.clone()
    uniform = [planner._sample_batched(batch_size).clone() for _ in range(2)]
    primary_after_uniform = primary_buffer._int_gen.get_state().clone()

    planner.reset_seed()
    mixed = [
        planner._generate_proposal_batch(batch_size, starts, goals, iteration).clone()
        for iteration in (0, 4)
    ]
    primary_after_mixed = primary_buffer._int_gen.get_state().clone()
    latent_after_mixed = latent_schedule.clone()

    for uniform_wave, mixed_wave in zip(uniform, mixed):
        torch.testing.assert_close(
            mixed_wave[:, planner.config.informed_sample_count :],
            uniform_wave[:, planner.config.informed_sample_count :],
            rtol=1.0e-5,
            atol=5.0e-7,
        )
    torch.testing.assert_close(primary_after_mixed, primary_after_uniform, rtol=0.0, atol=0.0)
    torch.testing.assert_close(latent_schedule, latent_initial)

    planner.reset_seed()
    torch.testing.assert_close(primary_buffer._int_gen.get_state(), primary_initial)
    repeated = [
        planner._generate_proposal_batch(batch_size, starts, goals, iteration).clone()
        for iteration in (0, 4)
    ]
    for actual, expected in zip(repeated, mixed):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        assert torch.all(actual >= lows)
        assert torch.all(actual <= highs)
    torch.testing.assert_close(latent_schedule, latent_after_mixed)
    assert set(planner._informed_proposal_executors) == {batch_size}
    planner.reset_cuda_graph()


def test_mixed_sampler_static_schedule_is_lane_and_wave_specific(
    resources: RRTPlannerResources,
):
    config = BiRRTConfig(
        samples_per_iteration=4,
        max_iterations=2,
        failure_extension_iterations=0,
        sampling_policy="mixed_ellipsoid",
        informed_sample_count=2,
        max_batched_lanes=4,
    )
    planner = BiRRTPlanner(resources, config)
    schedule = planner._informed_latent_schedule
    assert schedule is not None
    latent_rows = (
        config.informed_sample_count * (resources.action_dim + 1) + resources.action_dim - 1
    ) // resources.action_dim
    assert schedule.shape == (2, 4, latent_rows, resources.action_dim)
    assert not torch.equal(schedule[0, 0], schedule[0, 1])
    assert not torch.equal(schedule[0, 0], schedule[1, 0])

    overflow = config.max_batched_lanes + 1
    start = resources.action_bound_lows.view(1, -1).expand(overflow, -1)
    goal = resources.action_bound_highs.view(1, -1).expand(overflow, -1)
    with pytest.raises(ValueError, match="max_batched_lanes"):
        planner._generate_proposal_batch(overflow, start, goal, 0)
    planner.reset_cuda_graph()


def test_rrt_gpu_linear_interpolation_matches_resource_fallback(resources: RRTPlannerResources):
    planner = BiRRTPlanner(resources)
    path = torch.stack(
        (
            torch.zeros(resources.action_dim, device=resources.device_cfg.device),
            torch.linspace(-0.5, 0.5, resources.action_dim, device=resources.device_cfg.device),
            torch.ones(resources.action_dim, device=resources.device_cfg.device),
        )
    )
    expected = resources.get_interpolated_trajectory(
        [path],
        torch.ones(1, device=path.device, dtype=torch.bool),
        17,
        TrajInterpolationType.LINEAR,
    )
    actual = planner._interpolate_paths([path], 17, TrajInterpolationType.LINEAR)
    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)


def test_rrt_direct_and_obstructed_paths(resources: RRTPlannerResources):
    resources.reset_seed()
    samples = resources.uniform_sampler.generate_feasible_action_samples(256)
    starts = samples[:128]
    goals = samples[128:]
    blocked = _direct_blocked(resources, starts, goals)
    query_indices = [
        int(torch.where(~blocked)[0][0].item()),
        int(torch.where(blocked)[0][0].item()),
    ]
    planner = BiRRTPlanner(resources)
    returned_paths = []

    for query_index in query_indices:
        resources.reset_seed()
        result = planner.find_path(
            starts[query_index : query_index + 1],
            goals[query_index : query_index + 1],
            interpolate_waypoints=False,
            validate_interpolated_trajectory=False,
        )
        assert result.success.tolist() == [True]
        assert result.valid_query
        torch.testing.assert_close(result.plan_waypoints[0][0], starts[query_index])
        torch.testing.assert_close(result.plan_waypoints[0][-1], goals[query_index])
        assert _path_edges_valid(resources, result.plan_waypoints[0])
        returned_paths.append(result.plan_waypoints[0])

    workspace = planner._workspaces[1]
    assert workspace.path_reconstruction_executor.is_initialized
    assert len(workspace.finalization_executors) == 1
    assert next(iter(workspace.finalization_executors.values())).is_initialized
    torch.testing.assert_close(
        returned_paths[0],
        torch.stack((starts[query_indices[0]], goals[query_indices[0]])),
    )


def test_paired_planner_collects_every_direct_success_for_one_owner(
    resources: RRTPlannerResources,
):
    """Return every lane that connects in the first successful vectorized stage."""
    resources.reset_seed()
    feasible = resources.uniform_sampler.generate_feasible_action_samples(256)
    ambient = resources.uniform_sampler.generate_action_samples(1024, bounded=True)
    invalid_goals = ambient[~resources.check_samples_feasibility(ambient)]
    assert invalid_goals.shape[0] > 0
    start = feasible[0:1]
    candidates = feasible[1:]
    reachable = candidates[~_direct_blocked(resources, start.expand_as(candidates), candidates)][
        :3
    ]
    goals = torch.cat((invalid_goals[0:1], reachable), dim=0)
    starts = start.expand_as(goals).contiguous()

    result = BiRRTPlanner(resources).find_path(
        starts,
        goals,
        query_group_id=torch.zeros(goals.shape[0], device=goals.device, dtype=torch.int32),
        group_count=1,
        interpolate_waypoints=False,
        validate_interpolated_trajectory=False,
    )

    assert result.success.tolist() == [False, True, True, True]
    assert result.debug_info["growth_steps"] == 0
    assert result.debug_info["group_solved_mask"] == [True]
    assert result.plan_waypoints[0] is None
    for lane_index in range(1, goals.shape[0]):
        torch.testing.assert_close(result.plan_waypoints[lane_index][-1], goals[lane_index])


def test_query_mask_isolates_extreme_lane_from_direct_preprocessing(
    resources: RRTPlannerResources,
    monkeypatch: pytest.MonkeyPatch,
):
    """A masked extreme goal must not change a valid lane's direct result."""
    planner = BiRRTPlanner(
        resources,
        BiRRTConfig(shortcut_path=False),
    )
    device = resources.device_cfg.device
    dtype = resources.device_cfg.dtype
    dof = resources.action_dim
    starts = torch.zeros((2, dof), device=device, dtype=dtype)
    goals = torch.zeros_like(starts)
    goals[0, 0] = 0.1
    starts[1].fill_(-torch.inf)
    goals[1].fill_(torch.inf)
    original_starts = starts.clone()
    original_goals = goals.clone()
    staged_batches = []

    def globally_sensitive_feasibility(actions, **_kwargs):
        staged_batches.append(actions.clone())
        batch_is_finite = bool(torch.isfinite(actions).all().item())
        return torch.full(
            (actions.shape[0],),
            batch_is_finite,
            device=actions.device,
            dtype=torch.bool,
        )

    def globally_sensitive_direct(staged_starts, desired, **_kwargs):
        staged_batches.extend((staged_starts.clone(), desired.clone()))
        batch_is_finite = bool(
            (torch.isfinite(staged_starts).all() & torch.isfinite(desired).all()).item()
        )
        reached = torch.full(
            (2 * staged_starts.shape[0],),
            batch_is_finite,
            device=staged_starts.device,
            dtype=torch.bool,
        )
        endpoints = torch.cat((desired, staged_starts), dim=0)
        return endpoints, reached, reached.clone(), reached.clone()

    monkeypatch.setattr(resources, "check_samples_feasibility", globally_sensitive_feasibility)
    monkeypatch.setattr(curobo_runtime, "cuda_graphs", False)
    monkeypatch.setattr(
        planner,
        "_evaluate_staged_direct",
        lambda connector: globally_sensitive_direct(
            connector.request_starts,
            connector.request_goals,
        ),
    )
    result = planner.find_path(
        starts,
        goals,
        query_mask=torch.tensor([True, False], device=device),
        interpolate_waypoints=False,
        validate_interpolated_trajectory=False,
    )

    assert result.success.tolist() == [True, False]
    assert result.debug_info["growth_steps"] == 0
    assert all(bool(torch.isfinite(batch).all().item()) for batch in staged_batches)
    workspace = planner._workspaces[2]
    torch.testing.assert_close(workspace.start_tree.nodes[1, 0], starts[0])
    torch.testing.assert_close(workspace.goal_tree.nodes[1, 0], starts[0])
    assert torch.equal(starts, original_starts)
    assert torch.equal(goals, original_goals)


def test_normalized_request_retains_one_host_metadata_snapshot(
    resources: RRTPlannerResources,
) -> None:
    """Reuse one host copy of mask and group metadata throughout planning."""
    planner = BiRRTPlanner(resources)
    starts = torch.zeros(
        (3, resources.action_dim),
        **resources.device_cfg.as_torch_dict(),
    )
    goals = starts.clone()
    query_mask = torch.tensor([True, False, True], device=starts.device, dtype=torch.bool)
    query_group_id = torch.tensor([0, 0, 1], device=starts.device, dtype=torch.int32)

    request = planner._normalize_batched_request(
        starts,
        goals,
        None,
        query_mask,
        query_group_id,
        2,
    )

    assert request.query_mask_host == [True, False, True]
    assert request.group_ids_host == [0, 0, 1]
    assert request.has_requested_lanes


def test_single_owner_request_does_not_require_explicit_group_ids(
    resources: RRTPlannerResources,
) -> None:
    """Represent one completion owner without allocating caller-side group metadata."""
    planner = BiRRTPlanner(resources)
    starts = torch.zeros(
        (3, resources.action_dim),
        **resources.device_cfg.as_torch_dict(),
    )

    request = planner._normalize_batched_request(
        starts,
        starts.clone(),
        None,
        None,
        None,
        1,
    )

    assert request.group_count == 1
    assert request.group_ids_host == [0, 0, 0]
    assert request.goal_query_idx.tolist() == [0, 0, 0]


def test_all_masked_request_skips_preprocessing_and_workspace(
    resources: RRTPlannerResources,
    monkeypatch: pytest.MonkeyPatch,
):
    """Return an empty lane-shaped result without evaluating masked values."""
    planner = BiRRTPlanner(resources)
    device = resources.device_cfg.device
    dtype = resources.device_cfg.dtype
    starts = torch.full((2, resources.action_dim), -torch.inf, device=device, dtype=dtype)
    goals = torch.full_like(starts, torch.inf)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("all-masked requests must not run planner preprocessing")

    monkeypatch.setattr(resources, "check_samples_feasibility", unexpected)
    monkeypatch.setattr(planner, "_preprocess_search", unexpected)
    monkeypatch.setattr(planner, "_get_batched_workspace", unexpected)
    result = planner.find_path(
        starts,
        goals,
        query_mask=torch.zeros(2, device=device, dtype=torch.bool),
    )

    assert result.success.tolist() == [False, False]
    assert result.plan_waypoints == [None, None]
    assert torch.isinf(result.path_length).all()
    assert not result.valid_query
    assert result.debug_info["growth_steps"] == 0
    assert result.debug_info["group_valid_mask"] == [False, False]
    assert result.debug_info["group_solved_mask"] == [False, False]
    assert [query["skipped"] for query in result.debug_info["queries"]] == [
        "query masked",
        "query masked",
    ]


def test_interpolated_validation_reconciles_success_lengths_and_owner_metadata(
    resources: RRTPlannerResources,
    monkeypatch: pytest.MonkeyPatch,
):
    """Final trajectory validation must clear all success-derived metadata."""
    planner = BiRRTPlanner(
        resources,
        BiRRTConfig(shortcut_path=False),
    )
    device = resources.device_cfg.device
    dtype = resources.device_cfg.dtype
    dof = resources.action_dim
    interpolation_steps = 5
    starts = torch.zeros((2, dof), device=device, dtype=dtype)
    goals = torch.zeros_like(starts)
    goals[:, 0] = torch.tensor([0.1, 0.2], device=device, dtype=dtype)

    def staged_feasibility(actions, **_kwargs):
        feasible = torch.ones(actions.shape[0], device=actions.device, dtype=torch.bool)
        if actions.shape[0] == 2 * interpolation_steps:
            feasible[:interpolation_steps] = False
        return feasible

    def direct_success(_starts, desired, **_kwargs):
        reached = torch.ones(2 * desired.shape[0], device=desired.device, dtype=torch.bool)
        endpoints = torch.cat((desired, _starts), dim=0)
        return endpoints, reached, reached.clone(), torch.ones_like(reached)

    monkeypatch.setattr(resources, "check_samples_feasibility", staged_feasibility)
    monkeypatch.setattr(curobo_runtime, "cuda_graphs", False)
    monkeypatch.setattr(
        planner,
        "_evaluate_staged_direct",
        lambda connector: direct_success(
            connector.request_starts,
            connector.request_goals,
        ),
    )
    result = planner.find_path(
        starts,
        goals,
        interpolate_waypoints=True,
        interpolation_steps=interpolation_steps,
        validate_interpolated_trajectory=True,
    )

    assert result.success.tolist() == [False, True]
    assert torch.isinf(result.path_length[0])
    assert torch.isfinite(result.path_length[1])
    assert result.debug_info["group_solved_mask"] == [False, True]
    assert result.debug_info["interpolated_validation_failed_lanes"] == [0]
    assert result.debug_info["queries"][0]["interpolated_validation"] == "failed"


def test_paired_planner_collects_every_success_in_first_growth_step(
    resources: RRTPlannerResources,
    monkeypatch: pytest.MonkeyPatch,
):
    """Evaluate owner completion only after consuming the full step status vector."""
    device = resources.device_cfg.device
    dtype = resources.device_cfg.dtype
    dof = resources.action_dim
    lane_count = 3
    sample_count = 2
    planner = BiRRTPlanner(
        resources,
        BiRRTConfig(
            samples_per_iteration=sample_count,
            max_iterations=1,
            failure_extension_iterations=0,
            max_nodes_per_tree=128,
            shortcut_path=False,
        ),
    )
    samples = torch.zeros((lane_count, sample_count, dof), device=device, dtype=dtype)
    executed_samples = []
    workspace = _patch_fake_rrt_search(
        planner,
        resources,
        monkeypatch,
        samples,
        executed_samples,
    )
    step_calls = 0

    def execute_growth_step(_samples, _active, _env_query_idx, _goal_query_idx):
        nonlocal step_calls
        step_calls += 1
        no_connection = [sample_count, 0, 0, 1, 1]
        return [[0, 0, 0, 1, 1], no_connection, [1, 0, 0, 1, 1]]

    monkeypatch.setattr(workspace.growth_steps[0], "execute", execute_growth_step)
    starts = torch.zeros((lane_count, dof), device=device, dtype=dtype)
    goals = torch.ones_like(starts)

    result = planner.find_path(
        starts,
        goals,
        query_group_id=torch.zeros(lane_count, device=device, dtype=torch.int32),
        group_count=1,
        interpolate_waypoints=False,
        validate_interpolated_trajectory=False,
    )

    assert step_calls == 1
    assert result.debug_info["growth_steps"] == 1
    assert result.success.tolist() == [True, False, True]
    assert result.debug_info["group_solved_mask"] == [True]


def test_find_path_forwards_lane_scheduling_metadata(
    resources: RRTPlannerResources,
    monkeypatch: pytest.MonkeyPatch,
):
    """Keep paired-lane masks, owners, and environments available on the simple API."""
    planner = BiRRTPlanner(resources)
    starts = torch.zeros((3, resources.action_dim), device=resources.device_cfg.device)
    goals = torch.ones_like(starts)
    query_mask = torch.tensor([True, False, True], device=starts.device)
    query_group_id = torch.tensor([0, 0, 1], device=starts.device, dtype=torch.int32)
    env_query_idx = torch.zeros(3, device=starts.device, dtype=torch.int32)
    captured = {}
    sentinel = object()

    def fake_find_path_batched(flat_starts, flat_goals, **kwargs):
        captured["starts"] = flat_starts
        captured["goals"] = flat_goals
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(planner, "find_path_batched", fake_find_path_batched)
    result = planner.find_path(
        starts,
        goals,
        query_mask=query_mask,
        query_group_id=query_group_id,
        group_count=2,
        env_query_idx=env_query_idx,
        interpolate_waypoints=False,
        validate_interpolated_trajectory=False,
    )

    assert result is sentinel
    assert captured["starts"] is starts
    assert captured["goals"] is goals
    assert captured["kwargs"]["query_mask"] is query_mask
    assert captured["kwargs"]["query_group_id"] is query_group_id
    assert captured["kwargs"]["group_count"] == 2
    assert captured["kwargs"]["env_query_idx"] is env_query_idx
    assert "continue_after_success" not in captured["kwargs"]


def test_standard_find_path_forwards_per_query_environments(
    two_environment_resources: RRTPlannerResources,
):
    resources = two_environment_resources
    candidates = resources.uniform_sampler.generate_action_samples(64, bounded=True)
    empty_env = torch.ones(64, device=candidates.device, dtype=torch.int32)
    feasible = resources.check_samples_feasibility(candidates, env_query_idx=empty_env).reshape(-1)
    state = candidates[feasible][0]
    starts = state.view(1, -1).expand(2, -1).contiguous()

    result = BiRRTPlanner(resources).find_path(
        starts,
        starts.clone(),
        interpolate_waypoints=False,
        validate_interpolated_trajectory=False,
        env_query_idx=torch.tensor([0, 1], device=starts.device, dtype=torch.int32),
    )

    assert result.success.tolist() == [False, True]


@pytest.mark.parametrize("cuda_graphs_enabled", [False, True])
def test_growth_step_initialization_restores_trees_borrows_inputs_and_reinitializes(
    resources: RRTPlannerResources,
    monkeypatch: pytest.MonkeyPatch,
    cuda_graphs_enabled: bool,
):
    """Initialization may warm mutable trees, but exposes no logical tree side effects."""
    monkeypatch.setattr(curobo_runtime, "cuda_graphs", cuda_graphs_enabled)
    config = BiRRTConfig(
        samples_per_iteration=4,
        max_iterations=1,
        failure_extension_iterations=0,
        max_nodes_per_tree=64,
        fixed_steer_steps=8,
        shortcut_path=False,
    )
    planner = BiRRTPlanner(resources, config)
    workspace = planner._get_batched_workspace(2)
    growth_step = workspace.growth_steps[0]
    actions = resources.uniform_sampler.generate_feasible_action_samples(8)

    roots = (actions[:2], actions[2:4])
    for tree_index, (tree, tree_roots) in enumerate(
        zip((workspace.start_tree, workspace.goal_tree), roots)
    ):
        tree.nodes.copy_(tree_roots[:, None, :].expand(-1, config.max_nodes_per_tree, -1))
        tree.parents.fill_(-17 - tree_index)
        tree.parents[:, 0].fill_(-1)
        tree.parents[1, 1] = 0
        tree.counts.copy_(torch.tensor([1, 2], device=tree.counts.device, dtype=torch.int32))

    samples = torch.lerp(
        workspace.start_tree.nodes[:, :1],
        actions[4:8].view(1, config.samples_per_iteration, resources.action_dim),
        0.25,
    ).contiguous()
    active = torch.ones(2, device=samples.device, dtype=torch.bool)
    env_query_idx = torch.zeros(2, device=samples.device, dtype=torch.int32)
    initial_state = _snapshot_batched_tree_storage(workspace)

    goal_query_idx = torch.zeros_like(env_query_idx)
    growth_step.initialize(samples, active, env_query_idx, goal_query_idx)

    assert growth_step.is_initialized
    assert growth_step._executor._use_cuda_graph is cuda_graphs_enabled
    assert growth_step._executor._graph_output is not None
    assert len(growth_step._executor._graph_output) == 1
    assert growth_step._executor._graph_output[0].shape == (workspace.batch_size, 5)
    _assert_batched_tree_storage(workspace, initial_state)
    static_samples, static_active, static_env_query_idx, static_goal_query_idx = (
        growth_step._executor.static_inputs
    )
    assert static_samples is growth_step.active_nearest.desired
    assert static_active is growth_step.query_active
    assert static_env_query_idx is growth_step.env_query_idx
    assert static_goal_query_idx is growth_step.goal_query_idx
    assert static_samples.data_ptr() != samples.data_ptr()
    assert static_active.data_ptr() != active.data_ptr()
    assert static_env_query_idx.data_ptr() != env_query_idx.data_ptr()
    assert static_goal_query_idx.data_ptr() != goal_query_idx.data_ptr()
    torch.testing.assert_close(static_samples, samples, rtol=0.0, atol=0.0)
    assert torch.equal(static_active, active)
    assert torch.equal(static_env_query_idx, env_query_idx)
    assert torch.equal(static_goal_query_idx, goal_query_idx)

    status = growth_step.execute(samples, active, env_query_idx, goal_query_idx)
    assert len(status) == workspace.batch_size
    assert any(
        torch.any(tree.counts > reference).item()
        for tree, reference in zip(
            (workspace.start_tree, workspace.goal_tree),
            (initial_state[2], initial_state[5]),
        )
    )

    growth_step.reset()
    assert not growth_step.is_initialized
    with pytest.raises(ValueError, match="initialized"):
        _ = growth_step._executor.static_inputs

    workspace.start_tree.nodes[:, -1].fill_(0.125)
    workspace.goal_tree.parents[:, -1].fill_(-29)
    reinitialize_state = _snapshot_batched_tree_storage(workspace)
    reinitialize_samples = samples.flip(1).contiguous()
    growth_step.initialize(reinitialize_samples, active, env_query_idx, goal_query_idx)

    assert growth_step.is_initialized
    _assert_batched_tree_storage(workspace, reinitialize_state)
    reinitialized_static = growth_step._executor.static_inputs
    assert reinitialized_static[0] is growth_step.active_nearest.desired
    assert reinitialized_static[1] is growth_step.query_active
    assert reinitialized_static[2] is growth_step.env_query_idx
    assert reinitialized_static[3] is growth_step.goal_query_idx
    torch.testing.assert_close(reinitialized_static[0], reinitialize_samples, rtol=0.0, atol=0.0)
    growth_step.execute(reinitialize_samples, active, env_query_idx, goal_query_idx)
    growth_step.reset()


@pytest.mark.parametrize("projected", (False, True))
def test_workspace_reuses_fixed_growth_control_and_route_buffers(
    resources: RRTPlannerResources,
    projected: bool,
) -> None:
    """Share one fixed input ABI across both ordinary and projected growth orientations."""
    projection = RRTSampleProjectionConfig() if projected else None
    planner = BiRRTPlanner(
        resources,
        BiRRTConfig(shortcut_path=False, sample_projection=projection),
    )
    workspace = planner._get_batched_workspace(3)

    for growth_step in workspace.growth_steps:
        assert growth_step.query_active.data_ptr() == workspace.query_active.data_ptr()
        assert growth_step.env_query_idx.data_ptr() == workspace.env_query_idx.data_ptr()
        assert growth_step.goal_query_idx.data_ptr() == workspace.goal_query_idx.data_ptr()

    planner.reset_cuda_graph()


def test_batched_planner_solves_paired_obstructed_queries(resources: RRTPlannerResources):
    resources.reset_seed()
    samples = resources.uniform_sampler.generate_feasible_action_samples(256)
    starts = samples[:128]
    goals = samples[128:]
    indices = torch.where(_direct_blocked(resources, starts, goals))[0][:2]
    query_starts = starts[indices]
    query_goals = goals[indices]
    planner = BiRRTPlanner(resources)
    resources.reset_seed()

    result = planner.find_path_batched(
        query_starts,
        query_goals,
        interpolate_waypoints=False,
        validate_interpolated_trajectory=False,
    )

    assert result.success.tolist() == [True, True]
    assert result.debug_info["batch_bucket"] == 2
    assert result.debug_info["growth_steps"] > 0
    initialized_steps = [
        growth_step
        for workspace in planner._workspaces.values()
        for growth_step in workspace.growth_steps
        if growth_step.is_initialized
    ]
    assert initialized_steps
    for query_index, path in enumerate(result.plan_waypoints):
        torch.testing.assert_close(path[0], query_starts[query_index])
        torch.testing.assert_close(path[-1], query_goals[query_index])
        assert _path_edges_valid(resources, path)


def test_batched_planner_profiles_one_captured_phase(resources: RRTPlannerResources):
    resources.reset_seed()
    samples = resources.uniform_sampler.generate_feasible_action_samples(256)
    starts = samples[:128]
    goals = samples[128:]
    blocked_index = torch.where(_direct_blocked(resources, starts, goals))[0][0]
    planner = BiRRTPlanner(
        resources,
        BiRRTConfig(
            profile_cuda_events=True,
            profile_cuda_event_phase="active_feasibility",
        ),
    )
    resources.reset_seed()

    result = planner.find_path_batched(
        starts[blocked_index].view(1, -1),
        goals[blocked_index].view(1, -1),
        interpolate_waypoints=False,
        validate_interpolated_trajectory=False,
    )

    assert result.debug_info["growth_steps"] > 0
    assert result.debug_info["cuda_event_profile_phase"] == "active_feasibility"
    assert result.component_times is not None
    assert result.component_times["captured_active_feasibility"] > 0.0
    additive = {
        key: value
        for key, value in result.component_times.items()
        if not key.startswith("captured_")
    }
    assert sum(additive.values()) == pytest.approx(result.solve_time, abs=2.0e-6)


def test_batched_planner_measures_only_shortcut_time(resources: RRTPlannerResources):
    """Benchmark timing adds one shortcut interval without full profiling."""
    resources.reset_seed()
    samples = resources.uniform_sampler.generate_feasible_action_samples(256)
    starts = samples[:128]
    goals = samples[128:]
    blocked_index = torch.where(_direct_blocked(resources, starts, goals))[0][0]
    planner = BiRRTPlanner(
        resources,
        BiRRTConfig(measure_shortcut_time=True),
    )
    resources.reset_seed()

    result = planner.find_path_batched(
        starts[blocked_index].view(1, -1),
        goals[blocked_index].view(1, -1),
        interpolate_waypoints=False,
        validate_interpolated_trajectory=False,
    )

    assert result.component_times is not None
    assert set(result.component_times) == {"shortcut"}
    assert 0.0 <= result.component_times["shortcut"] <= result.solve_time


def test_batched_planner_masks_lanes_without_reducing_query_groups(resources: RRTPlannerResources):
    resources.reset_seed()
    samples = resources.uniform_sampler.generate_feasible_action_samples(256)
    start = samples[0:1]
    candidates = samples[1:]
    goals = candidates[~_direct_blocked(resources, start.expand_as(candidates), candidates)][:4]
    starts = start.expand_as(goals).contiguous()
    query_mask = torch.tensor([False, True, True, True], device=starts.device)
    query_group_id = torch.tensor([0, 0, 1, 1], device=starts.device, dtype=torch.int32)

    result = BiRRTPlanner(resources).find_path_batched(
        starts,
        goals,
        query_mask=query_mask,
        query_group_id=query_group_id,
        group_count=2,
        interpolate_waypoints=False,
        validate_interpolated_trajectory=False,
    )

    assert result.success.tolist() == [False, True, True, True]
    assert result.plan_waypoints[0] is None
    assert result.plan_waypoints[3] is not None
    assert result.debug_info["group_count"] == 2


def test_batched_planner_uses_owners_only_for_completion_scheduling(
    resources: RRTPlannerResources,
    monkeypatch: pytest.MonkeyPatch,
):
    device = resources.device_cfg.device
    dtype = resources.device_cfg.dtype
    dof = resources.action_dim
    sample_count = 4
    bucket = 4
    starts = torch.zeros((3, dof), device=device, dtype=dtype)
    goals = torch.zeros_like(starts)
    goals[0, 0] = 10.0
    goals[1, 0] = 1.0
    goals[2, 0] = 2.0
    planner = BiRRTPlanner(
        resources,
        BiRRTConfig(
            samples_per_iteration=sample_count,
            max_iterations=2,
            failure_extension_iterations=2,
            max_nodes_per_tree=128,
            shortcut_path=False,
        ),
    )

    def fake_tree():
        return SimpleNamespace(
            nodes=torch.zeros((bucket, 128, dof), device=device, dtype=dtype),
            parents=torch.full((bucket, 128), -1, device=device, dtype=torch.int64),
            counts=torch.ones(bucket, device=device, dtype=torch.int32),
        )

    execute_active = []
    scenario = {"last_group_connects": True}

    class FakeGrowthStep:
        is_initialized = True

        def execute(self, _samples, active, _env_query_idx, _goal_query_idx):
            execute_active.append(active.clone())
            no_connection = [sample_count, 0, 0, 1, 1]
            if not scenario["last_group_connects"]:
                active[:2] = False
                return [[0, 0, 0, 2, 2], [0, 0, 0, 2, 2], no_connection, no_connection]
            if len(execute_active) == 1:
                active[1] = False
                return [no_connection, [0, 0, 0, 2, 2], no_connection, no_connection]
            if len(execute_active) == 2:
                active[:3] = False
                return [[0, 0, 0, 2, 2], [0, 0, 0, 2, 2], [0, 0, 0, 2, 2], no_connection]
            raise AssertionError("growth continued after every valid group connected")

    path_reconstruction = SimpleNamespace(
        connection_indices=torch.full((bucket, 2), -1, device=device, dtype=torch.int64),
        output_paths=torch.empty((bucket, 16, dof), device=device, dtype=dtype),
        path_counts=torch.zeros(bucket, device=device, dtype=torch.int32),
        path_lengths=torch.full((bucket,), torch.inf, device=device, dtype=dtype),
    )
    workspace = SimpleNamespace(
        batch_size=bucket,
        start_tree=fake_tree(),
        goal_tree=fake_tree(),
        growth_steps=[FakeGrowthStep(), FakeGrowthStep()],
        query_active=torch.zeros(bucket, device=device, dtype=torch.bool),
        env_query_idx=torch.zeros(bucket, device=device, dtype=torch.int32),
        goal_query_idx=torch.zeros(bucket, device=device, dtype=torch.int32),
        path_reconstruction=path_reconstruction,
        path_counts_host=torch.zeros(bucket, dtype=torch.int32, pin_memory=True),
        shortcut_status_host=torch.zeros((bucket, 5), dtype=torch.int32, pin_memory=True),
        direct_status=torch.empty(bucket, device=device, dtype=torch.int32),
        direct_status_host=torch.empty(bucket, dtype=torch.int32, pin_memory=True),
        request_query_mask=torch.ones(bucket, device=device, dtype=torch.bool),
        request_query_mask_all_active=True,
        preprocessing_executors={},
    )

    monkeypatch.setattr(
        resources,
        "check_samples_feasibility",
        lambda samples, **_kwargs: torch.ones(samples.shape[0], device=device, dtype=torch.bool),
    )
    monkeypatch.setattr(planner, "_get_batched_workspace", lambda _batch_size: workspace)
    monkeypatch.setattr(
        planner,
        "_sample_batched",
        lambda _batch_size: torch.zeros((bucket, sample_count, dof), device=device, dtype=dtype),
    )

    def fake_direct_steer(_starts, desired, **_kwargs):
        reached = torch.zeros(2 * desired.shape[0], device=device, dtype=torch.bool)
        reached[[0, 3]] = True
        endpoints = torch.cat((desired, _starts), dim=0)
        return endpoints, reached, torch.zeros_like(reached), torch.ones_like(reached)

    def fake_reconstruct(_workspace, records):
        for record in records:
            record.path = torch.stack((starts[record.lane], goals[record.lane]))
            _workspace.path_reconstruction.path_lengths[record.lane] = torch.linalg.vector_norm(
                record.path[1] - record.path[0]
            )
            _workspace.path_reconstruction.path_counts[record.lane] = record.path.shape[0]

    monkeypatch.setattr(curobo_runtime, "cuda_graphs", False)
    monkeypatch.setattr(
        planner,
        "_evaluate_staged_direct",
        lambda connector: fake_direct_steer(
            connector.request_starts,
            connector.request_goals,
        ),
    )
    monkeypatch.setattr(planner, "_reconstruct_growth_paths", fake_reconstruct)

    result = planner.find_path_batched(
        starts,
        goals,
        query_group_id=torch.tensor([0, 0, 1], device=device, dtype=torch.int32),
        group_count=2,
        interpolate_waypoints=False,
        validate_interpolated_trajectory=False,
    )

    assert len(execute_active) == 2
    assert result.debug_info["growth_steps"] == 2
    assert [active.tolist() for active in execute_active] == [
        [False, True, True, False],
        [False, False, True, False],
    ]
    assert result.success.tolist() == [True, True, True]
    assert result.debug_info["group_solved_mask"] == [True, True]
    assert "selected_lane_ids" not in result.debug_info

    execute_active.clear()
    scenario["last_group_connects"] = False
    exhausted = planner.find_path_batched(
        starts,
        goals,
        query_group_id=torch.tensor([0, 0, 1], device=device, dtype=torch.int32),
        group_count=2,
        interpolate_waypoints=False,
        validate_interpolated_trajectory=False,
    )
    assert len(execute_active) == 4
    assert exhausted.success.tolist() == [True, True, False]
    assert exhausted.debug_info["growth_steps"] == 4
    assert exhausted.debug_info["budget_exhausted"]
    assert exhausted.debug_info["failure_extension_steps"] == 2


def test_growth_steps_alternate_start_and_goal_tree_orientations(
    resources: RRTPlannerResources,
    monkeypatch: pytest.MonkeyPatch,
):
    """Execute the two distinguishable orientations in strict round-robin order."""
    sample_count = 2
    planner = BiRRTPlanner(
        resources,
        BiRRTConfig(
            samples_per_iteration=sample_count,
            max_iterations=2,
            failure_extension_iterations=2,
            max_nodes_per_tree=128,
            goal_bias_count=0,
            shortcut_path=False,
        ),
    )
    device = resources.device_cfg.device
    dtype = resources.device_cfg.dtype
    dof = resources.action_dim
    orientations = []

    class FakeGrowthStep:
        is_initialized = True

        def __init__(self, orientation: int):
            self.orientation = orientation

        def execute(self, _samples, _active, _env_query_idx, _goal_query_idx):
            orientations.append(self.orientation)
            return [[sample_count, 0, 0, 1, 1]]

    roots = torch.zeros((1, 1, dof), device=device, dtype=dtype)
    workspace = SimpleNamespace(
        batch_size=1,
        start_tree=SimpleNamespace(nodes=roots),
        goal_tree=SimpleNamespace(nodes=roots.clone()),
        growth_steps=[FakeGrowthStep(0), FakeGrowthStep(1)],
        query_active=torch.ones(1, device=device, dtype=torch.bool),
    )
    request = SimpleNamespace(
        batch_size=1,
        x_start=torch.zeros((1, dof), device=device, dtype=dtype),
        route_goals=False,
    )
    state = SimpleNamespace(
        active_host=[True],
        group_solved=[False],
        group_valid_host=[True],
        group_ids_host=[0],
        connection_records=[],
    )
    monkeypatch.setattr(
        planner,
        "_generate_proposal_batch",
        lambda *_args: torch.zeros((1, sample_count, dof), device=device, dtype=dtype),
    )

    growth_steps, _, _ = planner._run_growth_steps(
        request,
        state,
        workspace,
        torch.zeros(1, device=device, dtype=torch.int32),
        torch.zeros(1, device=device, dtype=torch.int32),
        lambda _name: None,
    )

    assert growth_steps == 4
    assert orientations == [0, 1, 0, 1]


def test_growth_step_uses_direct_executor_when_cuda_graphs_are_disabled(
    resources: RRTPlannerResources,
    monkeypatch: pytest.MonkeyPatch,
):
    """Keep canonical growth-step execution functional when capture is globally disabled."""
    monkeypatch.setattr(curobo_runtime, "cuda_graphs", False)
    resources.reset_seed()
    samples = resources.uniform_sampler.generate_feasible_action_samples(256)
    starts = samples[:128]
    goals = samples[128:]
    blocked_index = torch.where(_direct_blocked(resources, starts, goals))[0][0]
    planner = BiRRTPlanner(
        resources,
        BiRRTConfig(
            max_iterations=1,
            failure_extension_iterations=0,
            shortcut_path=False,
        ),
    )
    resources.reset_seed()

    result = planner.find_path_batched(
        starts[blocked_index].view(1, -1),
        goals[blocked_index].view(1, -1),
        interpolate_waypoints=False,
        validate_interpolated_trajectory=False,
    )

    assert result.valid_query
    assert result.debug_info["growth_steps"] == 1
    initialized_steps = [
        growth_step
        for workspace in planner._workspaces.values()
        for growth_step in workspace.growth_steps
        if growth_step.is_initialized
    ]
    assert len(initialized_steps) == 1
    assert initialized_steps[0]._executor._use_cuda_graph is False
    assert initialized_steps[0]._executor._graph is None
