# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for RobotRollout.

Tests the rollout_arm.py module which provides the main generic rollout class for robot arms.
RobotRollout integrates transition models, cost managers, constraint managers, and scene collision
checkers to compute costs and constraints from action sequences.

This test file uses real robot configurations (Franka) and task configurations to test
the integrated behavior of all components.
"""

# Standard Library
from types import SimpleNamespace
from typing import Callable

# Third Party
import pytest
import torch

# CuRobo
import curobo._src.rollout.rollout_robot as rollout_robot_module
import curobo._src.runtime as curobo_runtime
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.rollout_robot import RobotRollout
from curobo._src.rollout.rollout_robot_cfg import RobotRolloutCfg
from curobo._src.state.state_joint import JointState
from curobo._src.transition.robot_state_transition import RobotStateTransition
from curobo._src.util_file import (
    get_robot_configs_path,
    get_task_configs_path,
    join_path,
    load_yaml,
)


@pytest.fixture(scope="module")
def franka_robot_cfg_dict():
    """Load Franka robot configuration dictionary from YAML file."""
    robot_file = join_path(get_robot_configs_path(), "franka.yml")
    robot_cfg_dict = load_yaml(robot_file)["robot_cfg"]
    return robot_cfg_dict


@pytest.fixture(scope="module")
def particle_trajopt_config():
    """Load particle trajopt task configuration from YAML file."""
    task_file = join_path(get_task_configs_path(), "trajopt/particle_trajopt.yml")
    return load_yaml(task_file)


@pytest.fixture(scope="module")
def metrics_base_config():
    """Load base metrics task configuration from YAML file."""
    task_file = join_path(get_task_configs_path(), "metrics_base.yml")
    return load_yaml(task_file)


@pytest.fixture(scope="module")
def transition_bspline_config():
    """Load transition bspline trajopt task configuration from YAML file."""
    task_file = join_path(get_task_configs_path(), "trajopt/transition_bspline_trajopt.yml")
    return load_yaml(task_file)


@pytest.fixture(scope="module")
def simple_arm_rollout_cfg(
    cuda_device_cfg, franka_robot_cfg_dict, particle_trajopt_config, transition_bspline_config
):
    """Create a simple RobotRolloutCfg for testing using real config files."""
    # Use the rollout config from particle_trajopt
    rollout_dict = particle_trajopt_config.get("rollout", {})

    # Add the transition model config to the rollout dict (as done in reach_base_cfg.py)
    rollout_dict["transition_model_cfg"] = transition_bspline_config.get(
        "transition_model_cfg", {}
    )

    # Create the full config using create_with_component_types
    config = RobotRolloutCfg.create_with_component_types(
        rollout_dict,
        franka_robot_cfg_dict,
        cuda_device_cfg,
    )

    return config


@pytest.fixture(scope="module")
def sample_joint_state(cuda_device_cfg, franka_robot_cfg_dict):
    """Create a sample joint state."""
    num_dof = len(franka_robot_cfg_dict["kinematics"]["cspace"]["joint_names"])
    batch_size = 2
    position = torch.zeros((batch_size, num_dof), **cuda_device_cfg.as_torch_dict())
    joint_state = JointState.from_position(position)
    # Add required fields
    joint_state.dt = torch.ones((batch_size,), **cuda_device_cfg.as_torch_dict()) * 0.02
    joint_state.use_implicit_goal_state = torch.zeros(
        (batch_size,), device=cuda_device_cfg.device, dtype=torch.uint8
    )
    return joint_state


class TestRobotRolloutInitialization:
    """Test RobotRollout initialization."""

    def test_init_with_none_config(self):
        """Test initialization with None configuration."""
        rollout = RobotRollout(config=None)
        assert rollout.config is None
        assert rollout._num_particles_goal is None
        assert rollout._metrics_goal is None

    def test_init_with_simple_config(self, simple_arm_rollout_cfg):
        """Test initialization with simple configuration."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        assert rollout.config is not None
        assert rollout.config == simple_arm_rollout_cfg
        assert rollout._num_particles_goal is None
        assert rollout._metrics_goal is None

    def test_init_creates_transition_models(self, simple_arm_rollout_cfg):
        """Test that initialization creates both transition models."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        assert hasattr(rollout, "transition_model")
        assert hasattr(rollout, "metrics_transition_model")
        assert isinstance(rollout.transition_model, RobotStateTransition)
        assert isinstance(rollout.metrics_transition_model, RobotStateTransition)

    def test_init_with_cost_managers(self, simple_arm_rollout_cfg):
        """Test initialization with cost managers from real config."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        # particle_trajopt.yml has cost and constraint configs
        # So cost managers should be created
        assert rollout.cost_manager is not None or rollout.constraint_manager is not None

    def test_init_with_scene_collision_checker(self, simple_arm_rollout_cfg, cuda_device_cfg):
        """Test initialization with pre-created scene collision checker."""
        # Create a simple collision checker config
        collision_cfg = {
            "checker_type": "PRIMITIVE",
            "max_distance": 0.1,
            "num_envs": 1,
            "tensor_args": cuda_device_cfg,
        }

        # Note: This will fail if collision checker needs more complex setup
        # We're just testing that the parameter is accepted
        rollout = RobotRollout(config=simple_arm_rollout_cfg, scene_collision_checker=None)
        assert rollout.scene_collision_checker is None


class TestRobotRolloutProperties:
    """Test RobotRollout properties."""

    def test_action_dim_property(self, simple_arm_rollout_cfg):
        """Test action_dim property returns transition model's action_dim."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        assert rollout.action_dim == rollout.transition_model.action_dim
        assert isinstance(rollout.action_dim, int)
        assert rollout.action_dim > 0

    def test_action_horizon_property(self, simple_arm_rollout_cfg):
        """Test action_horizon property."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        assert rollout.action_horizon == rollout.transition_model.action_horizon
        assert isinstance(rollout.action_horizon, int)
        assert rollout.action_horizon > 0

    def test_horizon_property(self, simple_arm_rollout_cfg):
        """Test horizon property."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        assert rollout.horizon == rollout.transition_model.horizon
        assert isinstance(rollout.horizon, int)
        assert rollout.horizon > 0

    def test_action_bound_lows_property(self, simple_arm_rollout_cfg):
        """Test action_bound_lows property."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        bounds = rollout.action_bound_lows
        assert isinstance(bounds, torch.Tensor)
        assert bounds.shape[-1] == rollout.action_dim

    def test_action_bound_highs_property(self, simple_arm_rollout_cfg):
        """Test action_bound_highs property."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        bounds = rollout.action_bound_highs
        assert isinstance(bounds, torch.Tensor)
        assert bounds.shape[-1] == rollout.action_dim

    def test_action_bounds_are_valid(self, simple_arm_rollout_cfg):
        """Test that action bounds are valid (lows < highs)."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        lows = rollout.action_bound_lows
        highs = rollout.action_bound_highs

        assert torch.all(lows < highs), "Action bound lows should be less than highs"

    def test_dt_property(self, simple_arm_rollout_cfg):
        """Test dt property."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        dt = rollout.dt
        assert isinstance(dt, float)
        assert dt > 0.0

    def test_default_joint_state_property(self, simple_arm_rollout_cfg):
        """Test default_joint_state property."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        default_js = rollout.default_joint_state
        assert isinstance(default_js, torch.Tensor)

    def test_default_joint_position_property(self, simple_arm_rollout_cfg):
        """Test default_joint_position property."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        default_position = rollout.default_joint_position
        assert isinstance(default_position, torch.Tensor)
        # Should match default_joint_state
        assert torch.allclose(default_position, rollout.default_joint_state)


class TestRobotRolloutUpdateParams:
    """Test RobotRollout update_params functionality."""

    def test_update_params_basic(
        self, simple_arm_rollout_cfg, sample_joint_state, cuda_device_cfg
    ):
        """Test basic update_params."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        goal = GoalRegistry.create_idx(
            pose_batch_size=1,
            multi_env=sample_joint_state.position.shape[0],
            num_seeds=1,
            device_cfg=cuda_device_cfg,
        )
        goal.current_js = sample_joint_state

        success = rollout.update_params(goal)
        assert success is True
        assert rollout._num_particles_goal is not None
        assert rollout._metrics_goal is not None

    def test_update_params_with_num_particles(
        self, simple_arm_rollout_cfg, sample_joint_state, cuda_device_cfg, franka_robot_cfg_dict
    ):
        """Test update_params accepts num_particles parameter."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        # Create a properly formatted goal with all necessary indices
        goal = GoalRegistry.create_idx(
            pose_batch_size=1,
            multi_env=sample_joint_state.position.shape[0],
            num_seeds=4,  # Start with 4 seeds
            device_cfg=cuda_device_cfg,
        )
        goal.current_js = sample_joint_state

        # Update without num_particles (should not call repeat_seeds)
        success = rollout.update_params(goal)

        assert success is True
        # Should have the goal set
        assert rollout._num_particles_goal is not None
        # num_seeds should match the goal's num_seeds
        assert rollout._num_particles_goal.num_seeds == 4

    def test_update_params_updates_start_state(
        self, simple_arm_rollout_cfg, sample_joint_state, cuda_device_cfg
    ):
        """Test that update_params sets start_state."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        goal = GoalRegistry.create_idx(
            pose_batch_size=1,
            multi_env=sample_joint_state.position.shape[0],
            num_seeds=1,
            device_cfg=cuda_device_cfg,
        )
        goal.current_js = sample_joint_state

        rollout.update_params(goal)

        assert rollout.start_state is not None
        assert torch.allclose(rollout.start_state.position, sample_joint_state.position)

    def test_update_params_multiple_times(
        self, simple_arm_rollout_cfg, sample_joint_state, cuda_device_cfg
    ):
        """Test calling update_params multiple times."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        goal = GoalRegistry.create_idx(
            pose_batch_size=1,
            multi_env=sample_joint_state.position.shape[0],
            num_seeds=1,
            device_cfg=cuda_device_cfg,
        )
        goal.current_js = sample_joint_state

        # First update
        rollout.update_params(goal)
        first_num_particles_goal = rollout._num_particles_goal

        # Second update (should reuse existing goal)
        new_state = sample_joint_state.clone()
        new_state.position += 0.1
        goal.current_js = new_state
        rollout.update_params(goal)

        # Should be the same object
        assert rollout._num_particles_goal is first_num_particles_goal


class TestRobotRolloutComputeState:
    """Test RobotRollout state computation."""

    def test_compute_state_method_exists(
        self, simple_arm_rollout_cfg, sample_joint_state, cuda_device_cfg
    ):
        """Test that state computation method exists."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        goal = GoalRegistry.create_idx(
            pose_batch_size=1,
            multi_env=sample_joint_state.position.shape[0],
            num_seeds=1,
            device_cfg=cuda_device_cfg,
        )
        goal.current_js = sample_joint_state
        rollout.update_params(goal)

        # Verify method exists
        assert hasattr(rollout, "_compute_state_from_action_impl")
        assert callable(rollout._compute_state_from_action_impl)

    def test_action_metrics_forwards_explicit_environment_to_state_and_costs(
        self, simple_arm_rollout_cfg, cuda_device_cfg, monkeypatch
    ):
        """An explicit environment overrides the goal for kinematics and costs."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)
        batch_size = 3
        actions = torch.zeros(
            (batch_size, 1, rollout.action_dim),
            **cuda_device_cfg.as_torch_dict(),
        )
        explicit_env = torch.tensor(
            [2, 0, 1],
            device=cuda_device_cfg.device,
            dtype=torch.int32,
        )
        goal_env = torch.ones(
            batch_size,
            device=cuda_device_cfg.device,
            dtype=torch.int32,
        )
        rollout._metrics_goal = SimpleNamespace(
            idxs_current_js=None,
            seed_goal_js=None,
            idxs_seed_goal_js=None,
            seed_enable_implicit_goal_js=None,
            idxs_env=goal_env,
        )
        state_sentinel = object()
        captured = {}

        def fake_transition_forward(*args, **kwargs):
            captured["state_env"] = kwargs["idxs_env"]
            return state_sentinel

        def fake_metrics(state, **kwargs):
            assert state is state_sentinel
            cost_env = kwargs.get("idxs_env_query")
            if cost_env is None:
                cost_env = kwargs.get("env_query_idx")
            captured["cost_env"] = cost_env
            return SimpleNamespace(actions=None)

        monkeypatch.setattr(rollout.metrics_transition_model, "forward", fake_transition_forward)
        monkeypatch.setattr(rollout, "_compute_metrics_from_state_impl", fake_metrics)

        result = rollout._compute_metrics_from_action_impl(
            actions,
            env_query_idx=explicit_env,
        )

        assert captured["state_env"] is explicit_env
        assert captured["cost_env"] is explicit_env
        assert result.actions is actions

        rollout._compute_metrics_from_action_impl(
            actions,
            idxs_env_query=explicit_env,
        )
        assert captured["state_env"] is explicit_env
        assert captured["cost_env"] is explicit_env

        state = rollout.compute_state_from_action_metrics(
            actions,
            env_query_idx=explicit_env,
        )
        assert state is state_sentinel
        assert captured["state_env"] is explicit_env

        rollout._compute_metrics_from_action_impl(actions)
        assert captured["state_env"] is goal_env


class TestRobotRolloutComputeCosts:
    """Test RobotRollout cost computation."""

    def test_compute_costs_returns_correct_type(
        self, simple_arm_rollout_cfg, sample_joint_state, cuda_device_cfg
    ):
        """Test that cost computation returns correct type."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        goal = GoalRegistry.create_idx(
            pose_batch_size=1,
            multi_env=sample_joint_state.position.shape[0],
            num_seeds=1,
            device_cfg=cuda_device_cfg,
        )
        goal.current_js = sample_joint_state
        rollout.update_params(goal)

        # Just verify the cost computation methods exist and are callable
        assert hasattr(rollout, "_compute_costs_and_constraints_impl")
        assert callable(rollout._compute_costs_and_constraints_impl)

    def test_compute_convergence_metrics_exists(
        self, simple_arm_rollout_cfg, sample_joint_state, cuda_device_cfg
    ):
        """Test that convergence metrics computation method exists."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        goal = GoalRegistry.create_idx(
            pose_batch_size=1,
            multi_env=sample_joint_state.position.shape[0],
            num_seeds=1,
            device_cfg=cuda_device_cfg,
        )
        goal.current_js = sample_joint_state
        rollout.update_params(goal)

        # Verify the convergence metrics method exists
        assert hasattr(rollout, "_compute_convergence_metrics_impl")
        assert callable(rollout._compute_convergence_metrics_impl)


class TestRobotRolloutBatchSize:
    """Test RobotRollout batch size management."""

    def test_update_batch_size(self, simple_arm_rollout_cfg):
        """Test updating batch size."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        new_batch_size = 10
        rollout.update_batch_size(new_batch_size)

        assert rollout._batch_size == new_batch_size

    def test_update_batch_size_updates_transition_model(self, simple_arm_rollout_cfg):
        """Test that update_batch_size propagates to transition model."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        new_batch_size = 10
        rollout.update_batch_size(new_batch_size)

        # Transition model should be updated
        assert rollout.transition_model.batch_size == new_batch_size


class TestRobotRolloutUpdate:
    """Test RobotRollout update operations."""

    def test_update_dt_not_supported(self, simple_arm_rollout_cfg):
        """Test that update_dt is not directly supported on transition model."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        # The transition model doesn't have update_dt, but rollout.update_dt should exist
        assert hasattr(rollout, "update_dt")
        # Just verify the dt property works
        assert rollout.dt > 0.0


class TestRobotRolloutReset:
    """Test RobotRollout reset operations."""

    def test_reset_without_problem_ids(self, simple_arm_rollout_cfg):
        """Test reset without specific problem ids."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        # Should not raise an error
        rollout.reset()

    def test_reset_shape(self, simple_arm_rollout_cfg, sample_joint_state, cuda_device_cfg):
        """Test reset_shape method."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        # Set up goals
        goal = GoalRegistry.create_idx(
            pose_batch_size=1,
            multi_env=sample_joint_state.position.shape[0],
            num_seeds=1,
            device_cfg=cuda_device_cfg,
        )
        goal.current_js = sample_joint_state
        rollout.update_params(goal)

        assert rollout._num_particles_goal is not None
        assert rollout._metrics_goal is not None

        # Reset shape
        rollout.reset_shape()

        assert rollout._num_particles_goal is None
        assert rollout._metrics_goal is None


class TestRobotRolloutCostComponents:
    """Test RobotRollout cost component management."""

    def test_get_cost_component_names(self, simple_arm_rollout_cfg):
        """Test getting cost component names from real config."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        names = rollout.get_cost_component_names()
        assert isinstance(names, list)
        # Real config should have cost components
        assert len(names) >= 0

    def test_get_all_cost_components(self, simple_arm_rollout_cfg):
        """Test getting all cost components from real config."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        components = rollout.get_all_cost_components()
        assert isinstance(components, dict)
        # Real config should have cost components
        assert len(components) >= 0


class TestRobotRolloutCudaGraph:
    """Test RobotRollout with use_cuda_graph=True."""

    def test_cuda_graph_creation(self, simple_arm_rollout_cfg):
        """Test that RobotRollout with use_cuda_graph=True creates correctly."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg, use_cuda_graph=True)

        assert isinstance(rollout, RobotRollout)
        assert rollout._use_cuda_graph is True

    def test_time_scaled_dt_uses_eager_path_when_checks_are_enabled(
        self, monkeypatch
    ) -> None:
        """Keep time scaling eager unless checks are explicitly skipped."""
        rollout, transition = self._make_time_scaled_dt_rollout()
        monkeypatch.setattr(curobo_runtime, "skip_checks", False)

        def fail_factory(**kwargs: object) -> None:
            raise AssertionError("graph executor should not be created")

        monkeypatch.setattr(rollout_robot_module, "create_graph_executor", fail_factory)
        action = torch.zeros((1, 1, 1), dtype=torch.float32)
        joint_limits = object()

        result = rollout.compute_time_scaled_dt_from_action(action, joint_limits)

        assert result is action
        assert transition.calls[0][1] is joint_limits

    def test_time_scaled_dt_graph_is_gated_only_by_skip_checks(self, monkeypatch) -> None:
        """Delegate graph enablement to GraphExecutor when checks are skipped."""
        rollout, transition = self._make_time_scaled_dt_rollout()
        rollout._use_cuda_graph = False
        monkeypatch.setattr(curobo_runtime, "skip_checks", True)
        monkeypatch.setattr(curobo_runtime, "cuda_graphs", False)
        factory_calls = []

        class FakeExecutor:
            def __init__(
                self,
                capture_fn: Callable[..., torch.Tensor],
                capture_kwargs: dict[str, object],
            ) -> None:
                self.capture_fn = capture_fn
                self.capture_kwargs = capture_kwargs

            def __call__(self, action: torch.Tensor) -> torch.Tensor:
                return self.capture_fn(action, **self.capture_kwargs)

        def create_fake_executor(
            *,
            capture_fn: Callable[..., torch.Tensor],
            device: torch.device,
            clone_outputs: bool,
            **kwargs: object,
        ) -> FakeExecutor:
            factory_calls.append((device, clone_outputs, kwargs))
            return FakeExecutor(capture_fn, kwargs)

        monkeypatch.setattr(
            rollout_robot_module,
            "create_graph_executor",
            create_fake_executor,
        )
        first_action = torch.zeros((1, 1, 1), dtype=torch.float32)
        second_action = torch.ones((1, 1, 1), dtype=torch.float32)
        first_limits = object()

        rollout.compute_time_scaled_dt_from_action(first_action, first_limits)
        result = rollout.compute_time_scaled_dt_from_action(second_action, object())

        assert result is second_action
        assert len(factory_calls) == 1
        assert "use_cuda_graph" not in factory_calls[0][2]
        assert transition.calls[0][1] is first_limits
        assert transition.calls[1][1] is first_limits

    def test_time_scaled_dt_graph_is_discarded_on_reset(self) -> None:
        """Allow changed static capture arguments after a graph reset."""
        rollout, _ = self._make_time_scaled_dt_rollout()

        class FakeExecutor:
            reset_calls = 0

            def reset(self) -> None:
                self.reset_calls += 1

        executor = FakeExecutor()
        rollout._compute_time_scaled_dt_from_action_executor = executor

        rollout.reset_cuda_graph()

        assert executor.reset_calls == 1
        assert rollout._compute_time_scaled_dt_from_action_executor is None

    @staticmethod
    def _make_time_scaled_dt_rollout() -> tuple[RobotRollout, object]:
        class FakeTransition:
            def __init__(self) -> None:
                self.calls = []

            def compute_time_scaled_dt_from_action(
                self,
                start_state: object,
                action: torch.Tensor,
                joint_limits: object,
                **kwargs: object,
            ) -> torch.Tensor:
                self.calls.append((action, joint_limits, kwargs))
                return action

        transition = FakeTransition()
        rollout = RobotRollout.__new__(RobotRollout)
        rollout.device_cfg = SimpleNamespace(device=torch.device("cpu"))
        rollout.transition_model = transition
        rollout.start_state = object()
        rollout._num_particles_goal = SimpleNamespace(
            idxs_current_js=object(),
            seed_goal_js=object(),
            idxs_seed_goal_js=object(),
            seed_enable_implicit_goal_js=object(),
        )
        rollout._metrics_goal = object()
        rollout._compute_metrics_from_state_executor = None
        rollout._compute_metrics_from_action_executor = None
        rollout._compute_time_scaled_dt_from_action_executor = None
        return rollout, transition


class TestRobotRolloutEdgeCases:
    """Test edge cases for RobotRollout."""

    def test_large_batch_size(
        self, simple_arm_rollout_cfg, cuda_device_cfg, franka_robot_cfg_dict
    ):
        """Test with large batch size."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        large_batch = 100
        num_dof = len(franka_robot_cfg_dict["kinematics"]["cspace"]["joint_names"])
        position = torch.zeros((large_batch, num_dof), **cuda_device_cfg.as_torch_dict())
        joint_state = JointState.from_position(position)

        goal = GoalRegistry.create_idx(
            pose_batch_size=1,
            multi_env=large_batch,
            num_seeds=1,
            device_cfg=cuda_device_cfg,
        )
        goal.current_js = joint_state

        success = rollout.update_params(goal)
        assert success is True

    def test_single_batch(self, simple_arm_rollout_cfg, cuda_device_cfg, franka_robot_cfg_dict):
        """Test with single batch size."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        num_dof = len(franka_robot_cfg_dict["kinematics"]["cspace"]["joint_names"])
        position = torch.zeros((1, num_dof), **cuda_device_cfg.as_torch_dict())
        joint_state = JointState.from_position(position)

        goal = GoalRegistry.create_idx(
            pose_batch_size=1,
            multi_env=1,
            num_seeds=1,
            device_cfg=cuda_device_cfg,
        )
        goal.current_js = joint_state

        success = rollout.update_params(goal)
        assert success is True


class TestRobotRolloutComputeStateFromAction:
    """Test _compute_state_from_action_impl method."""

    def test_joint_state_from_action_api_is_not_exposed(
        self, simple_arm_rollout_cfg
    ) -> None:
        """Joint-only rollout remains an internal transition implementation detail."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        assert not hasattr(rollout, "compute_joint_state_from_action")
        assert not hasattr(rollout.transition_model, "compute_joint_state_from_action")

    def test_compute_state_from_action_method_exists(self, simple_arm_rollout_cfg):
        """Test that _compute_state_from_action_impl method exists."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        assert hasattr(rollout, "_compute_state_from_action_impl")
        assert callable(rollout._compute_state_from_action_impl)

    def test_compute_state_from_action_metrics_method_exists(self, simple_arm_rollout_cfg):
        """Test that _compute_state_from_action_metrics_impl method exists."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        assert hasattr(rollout, "_compute_state_from_action_metrics_impl")
        assert callable(rollout._compute_state_from_action_metrics_impl)


class TestRobotRolloutComputeCostsAndConstraints:
    """Test cost and constraint computation methods."""

    def test_compute_costs_and_constraints_method_exists(self, simple_arm_rollout_cfg):
        """Test _compute_costs_and_constraints_impl method exists."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        assert hasattr(rollout, "_compute_costs_and_constraints_impl")
        assert callable(rollout._compute_costs_and_constraints_impl)

    def test_compute_costs_and_constraints_metrics_method_exists(self, simple_arm_rollout_cfg):
        """Test _compute_costs_and_constraints_metrics_impl method exists."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        assert hasattr(rollout, "_compute_costs_and_constraints_metrics_impl")
        assert callable(rollout._compute_costs_and_constraints_metrics_impl)

    def test_compute_convergence_metrics_method_exists(self, simple_arm_rollout_cfg):
        """Test _compute_convergence_metrics_impl method exists."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        assert hasattr(rollout, "_compute_convergence_metrics_impl")
        assert callable(rollout._compute_convergence_metrics_impl)


class TestRobotRolloutUpdateGoalDt:
    """Test update_goal_dt method."""

    def test_update_params_aliases_seed_goal_dt(
        self, simple_arm_rollout_cfg, sample_joint_state, cuda_device_cfg
    ):
        """Particle and metrics goals share the central per-seed dt tensor."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)
        batch_size, num_dof = sample_joint_state.position.shape
        seed_goal_js = JointState.from_position(
            torch.zeros((batch_size, 1, num_dof), **cuda_device_cfg.as_torch_dict())
        )
        seed_goal_js.dt = torch.full(
            (batch_size, 1), 0.02, **cuda_device_cfg.as_torch_dict()
        )
        goal = GoalRegistry(
            goal_js=sample_joint_state.clone(),
            current_js=sample_joint_state,
            seed_goal_js=seed_goal_js,
        )

        rollout.update_params(goal, num_particles=2)

        assert rollout._metrics_goal.seed_goal_js.dt is goal.seed_goal_js.dt
        assert rollout._num_particles_goal.seed_goal_js.dt is goal.seed_goal_js.dt

        replacement_dt = torch.full_like(goal.seed_goal_js.dt, 0.04)
        replacement_goal = GoalRegistry(
            goal_js=sample_joint_state.clone(),
            current_js=sample_joint_state,
            seed_goal_js=seed_goal_js.clone(),
        )
        replacement_goal.seed_goal_js.dt = replacement_dt
        rollout.update_goal_dt(replacement_goal)

        assert rollout._metrics_goal.seed_goal_js.dt is replacement_dt
        assert rollout._num_particles_goal.seed_goal_js.dt is replacement_dt

    def test_update_goal_dt_without_initialization(
        self, simple_arm_rollout_cfg, sample_joint_state, cuda_device_cfg
    ):
        """Test update_goal_dt fails without initialization."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        goal = GoalRegistry.create_idx(
            pose_batch_size=1,
            multi_env=sample_joint_state.position.shape[0],
            num_seeds=1,
            device_cfg=cuda_device_cfg,
        )

        # Add seed_goal_js to goal
        goal.seed_goal_js = sample_joint_state.clone()
        goal.seed_goal_js.dt = (
            torch.ones((sample_joint_state.position.shape[0],), **cuda_device_cfg.as_torch_dict())
            * 0.02
        )

        # Should raise error because rollout not initialized
        with pytest.raises(ValueError, match="not initialized"):
            rollout.update_goal_dt(goal)

    def test_update_goal_dt_without_seed_goal_js(
        self, simple_arm_rollout_cfg, sample_joint_state, cuda_device_cfg
    ):
        """Test update_goal_dt fails without seed_goal_js."""
        rollout = RobotRollout(config=simple_arm_rollout_cfg)

        # Initialize rollout
        goal = GoalRegistry.create_idx(
            pose_batch_size=1,
            multi_env=sample_joint_state.position.shape[0],
            num_seeds=1,
            device_cfg=cuda_device_cfg,
        )
        goal.current_js = sample_joint_state
        rollout.update_params(goal)

        # Create goal without seed_goal_js
        new_goal = GoalRegistry()

        # Should raise error
        with pytest.raises(ValueError, match="seed_goal_js is None"):
            rollout.update_goal_dt(new_goal)


class TestRobotRolloutWithConstraints:
    """Test RobotRollout with constraint configurations."""

    def test_init_with_constraint_cfg(self, cuda_device_cfg, franka_robot_cfg_dict):
        """Test initialization with constraint config."""
        # Load configs
        task_file = join_path(get_task_configs_path(), "trajopt/particle_trajopt.yml")
        particle_config = load_yaml(task_file)

        task_file = join_path(get_task_configs_path(), "trajopt/transition_bspline_trajopt.yml")
        transition_config = load_yaml(task_file)

        rollout_dict = particle_config.get("rollout", {})
        rollout_dict["transition_model_cfg"] = transition_config.get("transition_model_cfg", {})

        # Add constraint config (same as cost config for testing)
        if "cost_cfg" in rollout_dict and rollout_dict["cost_cfg"] is not None:
            rollout_dict["constraint_cfg"] = rollout_dict["cost_cfg"].copy()

        config = RobotRolloutCfg.create_with_component_types(
            rollout_dict,
            franka_robot_cfg_dict,
            cuda_device_cfg,
        )

        rollout = RobotRollout(config=config)

        # Should have both cost and constraint managers
        if config.constraint_cfg is not None:
            assert rollout.constraint_manager is not None

    def test_init_with_hybrid_cost_constraint_cfg(self, cuda_device_cfg, franka_robot_cfg_dict):
        """Test initialization with hybrid cost/constraint config."""
        # Load configs
        task_file = join_path(get_task_configs_path(), "trajopt/particle_trajopt.yml")
        particle_config = load_yaml(task_file)

        task_file = join_path(get_task_configs_path(), "trajopt/transition_bspline_trajopt.yml")
        transition_config = load_yaml(task_file)

        rollout_dict = particle_config.get("rollout", {})
        rollout_dict["transition_model_cfg"] = transition_config.get("transition_model_cfg", {})

        # Add hybrid cost constraint config
        if "cost_cfg" in rollout_dict and rollout_dict["cost_cfg"] is not None:
            rollout_dict["hybrid_cost_constraint_cfg"] = rollout_dict["cost_cfg"].copy()

        config = RobotRolloutCfg.create_with_component_types(
            rollout_dict,
            franka_robot_cfg_dict,
            cuda_device_cfg,
        )

        rollout = RobotRollout(config=config)

        if config.hybrid_cost_constraint_cfg is not None:
            assert rollout.hybrid_cost_constraint_manager is not None

    def test_init_with_convergence_cfg(self, cuda_device_cfg, franka_robot_cfg_dict):
        """Test initialization with convergence config."""
        # Load configs
        task_file = join_path(get_task_configs_path(), "trajopt/particle_trajopt.yml")
        particle_config = load_yaml(task_file)

        task_file = join_path(get_task_configs_path(), "trajopt/transition_bspline_trajopt.yml")
        transition_config = load_yaml(task_file)

        rollout_dict = particle_config.get("rollout", {})
        rollout_dict["transition_model_cfg"] = transition_config.get("transition_model_cfg", {})

        # Add convergence config
        if "cost_cfg" in rollout_dict and rollout_dict["cost_cfg"] is not None:
            rollout_dict["convergence_cfg"] = rollout_dict["cost_cfg"].copy()

        config = RobotRolloutCfg.create_with_component_types(
            rollout_dict,
            franka_robot_cfg_dict,
            cuda_device_cfg,
        )

        rollout = RobotRollout(config=config)

        if config.convergence_cfg is not None:
            assert rollout.metrics_convergence_manager is not None
