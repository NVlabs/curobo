# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RobotCostManager.

Tests the standalone, flat cost manager used by the robot rollout. It owns
the full set of joint-space, task-space, collision, and constraint costs with
no inheritance.
"""

# Third Party
import pytest
import torch
import warp as wp

# CuRobo
from curobo._src.cost.cost_cspace_cfg import CSpaceCostCfg
from curobo._src.cost.cost_cspace_dist_cfg import CSpaceDistCostCfg
from curobo._src.cost.cost_cspace_type import CSpaceCostType
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.metrics import CostCollection
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState


@pytest.fixture
def franka_robot_cfg_dict():
    from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml

    robot_file = join_path(get_robot_configs_path(), "franka.yml")
    robot_cfg_dict = load_yaml(robot_file)["robot_cfg"]
    return robot_cfg_dict


@pytest.fixture
def transition_ik_config():
    from curobo._src.util_file import get_task_configs_path, join_path, load_yaml

    task_file = join_path(get_task_configs_path(), "ik/transition_ik.yml")
    return load_yaml(task_file)


@pytest.fixture
def franka_transition_model(cuda_device_cfg, franka_robot_cfg_dict, transition_ik_config):
    from curobo._src.transition.robot_state_transition import RobotStateTransition
    from curobo._src.transition.robot_state_transition_cfg import RobotStateTransitionCfg

    transition_dict = transition_ik_config.get("transition_model_cfg", {})
    transition_dict["horizon"] = 10
    transition_dict["batch_size"] = 4

    transition_cfg = RobotStateTransitionCfg.create(
        transition_dict, franka_robot_cfg_dict, cuda_device_cfg
    )
    return RobotStateTransition(transition_cfg)


@pytest.fixture
def sample_goal_registry(cuda_device_cfg, franka_transition_model):
    batch_size = 4
    dof = franka_transition_model.num_dof

    goal_position = torch.zeros(
        batch_size, dof, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
    )
    goal_js = JointState(position=goal_position, device_cfg=cuda_device_cfg)

    current_position = torch.zeros(
        batch_size, dof, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
    )
    current_velocity = torch.zeros(
        batch_size, dof, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
    )
    current_js = JointState(
        position=current_position, velocity=current_velocity, device_cfg=cuda_device_cfg
    )

    idxs_goal_js = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
    idxs_current_js = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)

    return GoalRegistry(
        goal_js=goal_js,
        idxs_goal_js=idxs_goal_js,
        current_js=current_js,
        idxs_current_js=idxs_current_js,
    )


@pytest.fixture
def sample_robot_state(cuda_device_cfg, franka_transition_model):
    batch_size = 4
    horizon = 10
    dof = franka_transition_model.num_dof

    position = torch.randn(
        batch_size, horizon, dof, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
    )
    velocity = torch.zeros(
        batch_size, horizon, dof, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
    )
    acceleration = torch.zeros(
        batch_size, horizon, dof, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
    )
    jerk = torch.zeros(
        batch_size, horizon, dof, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
    )

    joint_state = JointState(
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        jerk=jerk,
        device_cfg=cuda_device_cfg,
    )
    return RobotState(joint_state=joint_state)


class TestRobotCostManagerInitialization:
    def test_default_initialization(self, cuda_device_cfg):
        manager = RobotCostManager(device_cfg=cuda_device_cfg)

        assert manager.costs == {}
        assert manager.device_cfg.device == cuda_device_cfg.device
        assert manager._initialized is False

    def test_no_inheritance(self):
        assert RobotCostManager.__bases__ == (object,)


class TestRobotCostManagerInitializeFromConfig:
    def test_initialize_with_empty_config(self, cuda_device_cfg, franka_transition_model):
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        config = RobotCostManagerCfg()

        manager.initialize_from_config(config, franka_transition_model)

        assert manager._initialized is True
        assert len(manager.costs) == 0

    def test_initialize_with_start_cspace_dist(self, cuda_device_cfg, franka_transition_model):
        config = RobotCostManagerCfg(
            start_cspace_dist_cfg=CSpaceDistCostCfg(weight=1.0, device_cfg=cuda_device_cfg)
        )
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        assert manager.has_cost("start_cspace_dist")

    def test_initialize_with_target_cspace_dist(self, cuda_device_cfg, franka_transition_model):
        config = RobotCostManagerCfg(
            target_cspace_dist_cfg=CSpaceDistCostCfg(weight=0.5, device_cfg=cuda_device_cfg)
        )
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        assert manager.has_cost("target_cspace_dist")

    def test_initialize_with_tool_pose(self, cuda_device_cfg, franka_transition_model):
        wp.init()
        config = RobotCostManagerCfg(
            tool_pose_cfg=ToolPoseCostCfg(weight=[1.0, 1.0], device_cfg=cuda_device_cfg)
        )
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        assert manager.has_cost("tool_pose")

    def test_initialize_with_cspace(self, cuda_device_cfg, franka_transition_model):
        config = RobotCostManagerCfg(
            cspace_cfg=CSpaceCostCfg(
                weight=[1.0, 1.0, 1.0, 1.0, 1.0],
                device_cfg=cuda_device_cfg,
                activation_distance=[0.0, 0.0, 0.0, 0.0, 0.0],
                cost_type=CSpaceCostType.STATE,
            )
        )
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        assert manager.has_cost("cspace")

    def test_initialize_multiple_costs(self, cuda_device_cfg, franka_transition_model):
        config = RobotCostManagerCfg(
            start_cspace_dist_cfg=CSpaceDistCostCfg(weight=1.0, device_cfg=cuda_device_cfg),
            target_cspace_dist_cfg=CSpaceDistCostCfg(weight=0.5, device_cfg=cuda_device_cfg),
            cspace_cfg=CSpaceCostCfg(
                weight=[1.0, 1.0, 1.0, 1.0, 1.0],
                device_cfg=cuda_device_cfg,
                activation_distance=[0.0, 0.0, 0.0, 0.0, 0.0],
                cost_type=CSpaceCostType.STATE,
            ),
        )
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        assert manager.has_cost("start_cspace_dist")
        assert manager.has_cost("target_cspace_dist")
        assert manager.has_cost("cspace")


class TestRobotCostManagerComputeCosts:
    def test_compute_costs_empty(
        self, cuda_device_cfg, franka_transition_model, sample_robot_state
    ):
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        config = RobotCostManagerCfg()
        manager.initialize_from_config(config, franka_transition_model)

        result = manager.compute_costs(sample_robot_state)

        assert isinstance(result, CostCollection)
        assert result.is_empty()

    def test_compute_costs_without_goal(
        self, cuda_device_cfg, franka_transition_model, sample_robot_state
    ):
        config = RobotCostManagerCfg(
            start_cspace_dist_cfg=CSpaceDistCostCfg(weight=1.0, device_cfg=cuda_device_cfg),
            cspace_cfg=CSpaceCostCfg(
                weight=[1.0, 1.0, 1.0, 1.0, 1.0],
                device_cfg=cuda_device_cfg,
                activation_distance=[0.0, 0.0, 0.0, 0.0, 0.0],
                cost_type=CSpaceCostType.STATE,
            ),
        )
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        result = manager.compute_costs(sample_robot_state, goal=None)

        assert isinstance(result, CostCollection)
        assert "cspace" in result.names

    def test_compute_costs_with_goal(
        self, cuda_device_cfg, franka_transition_model, sample_robot_state, sample_goal_registry
    ):
        config = RobotCostManagerCfg(
            cspace_cfg=CSpaceCostCfg(
                weight=[1.0, 1.0, 1.0, 1.0, 1.0],
                device_cfg=cuda_device_cfg,
                activation_distance=[0.0, 0.0, 0.0, 0.0, 0.0],
                cost_type=CSpaceCostType.STATE,
            ),
        )
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        result = manager.compute_costs(sample_robot_state, goal=sample_goal_registry)

        assert isinstance(result, CostCollection)
        assert "cspace" in result.names


class TestRobotCostManagerComputeConvergence:
    def test_convergence_empty(
        self, cuda_device_cfg, franka_transition_model, sample_robot_state
    ):
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        config = RobotCostManagerCfg()
        manager.initialize_from_config(config, franka_transition_model)

        result = manager.compute_convergence(sample_robot_state, goal=None)

        assert isinstance(result, CostCollection)
        assert result.is_empty()

    def test_convergence_with_start_cspace_dist(
        self, cuda_device_cfg, franka_transition_model, sample_goal_registry
    ):
        config = RobotCostManagerCfg(
            start_cspace_dist_cfg=CSpaceDistCostCfg(weight=1.0, device_cfg=cuda_device_cfg)
        )
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        batch_size = 4
        horizon = 10
        dof = franka_transition_model.num_dof
        joint_state = JointState(
            position=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            velocity=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            acceleration=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            jerk=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            device_cfg=cuda_device_cfg,
        )
        robot_state = RobotState(joint_state=joint_state)

        result = manager.compute_convergence(robot_state, goal=sample_goal_registry)

        assert isinstance(result, CostCollection)
        assert "start_cspace_dist_tolerance" in result.names

    def test_convergence_with_target_cspace_dist(
        self, cuda_device_cfg, franka_transition_model, sample_goal_registry
    ):
        config = RobotCostManagerCfg(
            target_cspace_dist_cfg=CSpaceDistCostCfg(weight=1.0, device_cfg=cuda_device_cfg)
        )
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        batch_size = 4
        horizon = 10
        dof = franka_transition_model.num_dof
        joint_state = JointState(
            position=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            velocity=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            acceleration=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            jerk=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            device_cfg=cuda_device_cfg,
        )
        robot_state = RobotState(joint_state=joint_state)

        result = manager.compute_convergence(robot_state, goal=sample_goal_registry)

        assert isinstance(result, CostCollection)
        assert "target_cspace_dist_tolerance" in result.names

    def test_convergence_with_both_cspace_dist(
        self, cuda_device_cfg, franka_transition_model, sample_goal_registry
    ):
        config = RobotCostManagerCfg(
            start_cspace_dist_cfg=CSpaceDistCostCfg(weight=1.0, device_cfg=cuda_device_cfg),
            target_cspace_dist_cfg=CSpaceDistCostCfg(weight=0.5, device_cfg=cuda_device_cfg),
        )
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        batch_size = 4
        horizon = 10
        dof = franka_transition_model.num_dof
        joint_state = JointState(
            position=torch.randn(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            velocity=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            acceleration=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            jerk=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            device_cfg=cuda_device_cfg,
        )
        robot_state = RobotState(joint_state=joint_state)
        manager.setup_batch_tensors(batch_size, horizon)

        result = manager.compute_convergence(robot_state, goal=sample_goal_registry)

        assert isinstance(result, CostCollection)
        assert "start_cspace_dist_tolerance" in result.names
        assert "target_cspace_dist_tolerance" in result.names


class TestRobotCostManagerUpdateParams:
    def test_update_params_not_initialized(self, cuda_device_cfg):
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.update_params(dt=0.1)
        assert manager._initialized is False

    def test_update_params_with_dt(self, cuda_device_cfg, franka_transition_model):
        config = RobotCostManagerCfg(
            start_cspace_dist_cfg=CSpaceDistCostCfg(weight=1.0, device_cfg=cuda_device_cfg)
        )
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        manager.update_params(dt=0.05)

        assert manager.get_cost("start_cspace_dist")._dt == 0.05

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_update_params_tool_pose_criteria_invalid(
        self, cuda_device_cfg, franka_transition_model
    ):
        config = RobotCostManagerCfg(
            tool_pose_cfg=ToolPoseCostCfg(weight=[1.0, 1.0], device_cfg=cuda_device_cfg)
        )
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        try:
            manager.update_params(tool_pose_criteria="invalid")
        except ValueError as e:
            assert "must be a dict" in str(e)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_update_params_tool_pose_criteria_valid(
        self, cuda_device_cfg, franka_transition_model
    ):
        config = RobotCostManagerCfg(
            tool_pose_cfg=ToolPoseCostCfg(weight=[1.0, 1.0], device_cfg=cuda_device_cfg)
        )
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        manager.update_params(tool_pose_criteria={})


class TestRobotCostManagerGradients:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_multiple_costs_gradients(
        self, cuda_device_cfg, franka_transition_model, sample_goal_registry
    ):
        config = RobotCostManagerCfg(
            start_cspace_dist_cfg=CSpaceDistCostCfg(
                weight=1.0, device_cfg=cuda_device_cfg, use_grad_input=True
            ),
            cspace_cfg=CSpaceCostCfg(
                weight=[1.0, 1.0, 1.0, 1.0, 1.0],
                device_cfg=cuda_device_cfg,
                activation_distance=[0.0, 0.0, 0.0, 0.0, 0.0],
                cost_type=CSpaceCostType.STATE,
            ),
        )
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        batch_size = 4
        horizon = 10
        dof = franka_transition_model.num_dof

        position = torch.randn(
            batch_size, horizon, dof, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
        ).requires_grad_(True)

        joint_state = JointState(
            position=position,
            velocity=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            acceleration=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            jerk=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            device_cfg=cuda_device_cfg,
        )
        robot_state = RobotState(joint_state=joint_state)
        manager.setup_batch_tensors(batch_size, horizon)

        result = manager.compute_costs(robot_state, goal=sample_goal_registry)

        assert "cspace" in result.names

        total_cost = torch.tensor(0.0, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        for cost in result.values:
            total_cost = total_cost + cost.sum()
        total_cost.backward()

        assert position.grad is not None


class TestRobotCostManagerDisableEnable:
    def test_disable_cost(self, cuda_device_cfg, franka_transition_model):
        config = RobotCostManagerCfg(
            start_cspace_dist_cfg=CSpaceDistCostCfg(weight=1.0, device_cfg=cuda_device_cfg)
        )
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        manager.disable_cost_component("start_cspace_dist")

        assert manager.get_cost("start_cspace_dist").enabled is False

    def test_enable_cost(self, cuda_device_cfg, franka_transition_model):
        config = RobotCostManagerCfg(
            start_cspace_dist_cfg=CSpaceDistCostCfg(weight=1.0, device_cfg=cuda_device_cfg)
        )
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        manager.disable_cost_component("start_cspace_dist")
        manager.enable_cost_component("start_cspace_dist")

        assert manager.get_cost("start_cspace_dist").enabled is True

    def test_get_enabled_costs(self, cuda_device_cfg, franka_transition_model):
        config = RobotCostManagerCfg(
            start_cspace_dist_cfg=CSpaceDistCostCfg(weight=1.0, device_cfg=cuda_device_cfg),
            target_cspace_dist_cfg=CSpaceDistCostCfg(weight=0.5, device_cfg=cuda_device_cfg),
        )
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        manager.disable_cost_component("target_cspace_dist")

        enabled = manager.get_enabled_costs()
        assert "start_cspace_dist" in enabled
        assert "target_cspace_dist" not in enabled


class TestRobotCostManagerGetCostComponents:
    def test_get_cost_component_names(self, cuda_device_cfg, franka_transition_model):
        config = RobotCostManagerCfg(
            start_cspace_dist_cfg=CSpaceDistCostCfg(weight=1.0, device_cfg=cuda_device_cfg),
            target_cspace_dist_cfg=CSpaceDistCostCfg(weight=0.5, device_cfg=cuda_device_cfg),
        )
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        names = manager.get_cost_component_names()
        assert "start_cspace_dist" in names
        assert "target_cspace_dist" in names

    def test_get_cost_components(self, cuda_device_cfg, franka_transition_model):
        config = RobotCostManagerCfg(
            start_cspace_dist_cfg=CSpaceDistCostCfg(weight=1.0, device_cfg=cuda_device_cfg)
        )
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        components = manager.get_cost_components()
        assert "start_cspace_dist" in components
        assert components["start_cspace_dist"] is not None

    def test_get_cost_returns_none_for_missing(self, cuda_device_cfg):
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        assert manager.get_cost("nonexistent") is None

    def test_has_cost_false_for_missing(self, cuda_device_cfg):
        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        assert manager.has_cost("nonexistent") is False
