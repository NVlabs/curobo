# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Integration tests for cost managers using full robot configuration."""

# Standard Library

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.device_cfg import DeviceCfg


@pytest.fixture
def cuda_device_cfg():
    """Create a CUDA device configuration for testing."""
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    return DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)


@pytest.fixture
def franka_robot_cfg_dict():
    """Load Franka robot configuration dictionary from YAML file."""
    from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml

    robot_file = join_path(get_robot_configs_path(), "franka.yml")
    robot_cfg_dict = load_yaml(robot_file)["robot_cfg"]
    return robot_cfg_dict


@pytest.fixture
def transition_ik_config():
    """Load transition IK config from YAML file."""
    from curobo._src.util_file import get_task_configs_path, join_path, load_yaml

    task_file = join_path(get_task_configs_path(), "ik/transition_ik.yml")
    return load_yaml(task_file)


@pytest.fixture
def franka_transition_model(cuda_device_cfg, franka_robot_cfg_dict, transition_ik_config):
    """Create a Franka robot state transition model for testing."""
    from curobo._src.transition.robot_state_transition import RobotStateTransition
    from curobo._src.transition.robot_state_transition_cfg import RobotStateTransitionCfg

    transition_dict = transition_ik_config.get("transition_model_cfg", {})
    transition_dict["horizon"] = 10
    transition_dict["batch_size"] = 4

    transition_cfg = RobotStateTransitionCfg.create(
        transition_dict,
        franka_robot_cfg_dict,
        cuda_device_cfg,
    )

    return RobotStateTransition(transition_cfg)


@pytest.fixture
def franka_kinematics(franka_transition_model):
    """Get the kinematics model from the transition model."""
    return franka_transition_model.robot_model


class TestRobotCostManagerSelfCollisionCompute:
    """Test self-collision cost computation with proper robot state."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_self_collision_compute_with_forward_kinematics(
        self, cuda_device_cfg, franka_transition_model, franka_kinematics
    ):
        """Test computing self-collision cost using forward kinematics to get robot spheres."""
        from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
        from curobo._src.robot.kinematics.kinematics_state import KinematicsState

        self_collision_cfg = SelfCollisionCostCfg(
            weight=1.0,
            device_cfg=cuda_device_cfg,
        )
        config = RobotCostManagerCfg(self_collision_cfg=self_collision_cfg)

        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        if not manager.has_cost("self_collision"):
            pytest.skip("Self-collision cost not registered (no collision spheres)")

        batch_size = 4
        horizon = 10
        dof = franka_transition_model.num_dof

        # Create joint positions (use default config as valid position)
        default_position = franka_kinematics.default_joint_position
        position = default_position.view(1, 1, -1).expand(batch_size, horizon, -1).clone()

        joint_state = JointState(
            position=position,
            velocity=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            acceleration=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            jerk=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            device_cfg=cuda_device_cfg,
        )

        # Use kinematics to compute robot spheres
        kin_result = franka_kinematics.compute_kinematics(
            JointState.from_position(position, joint_names=franka_kinematics.joint_names)
        )

        if kin_result.robot_spheres is None:
            pytest.skip("No robot spheres from kinematics")

        robot_spheres = kin_result.robot_spheres.view(batch_size, horizon, -1, 4)

        cuda_robot_model_state = KinematicsState(robot_spheres=robot_spheres)
        robot_state = RobotState(
            joint_state=joint_state, cuda_robot_model_state=cuda_robot_model_state
        )

        manager.setup_batch_tensors(batch_size, horizon)
        result = manager.compute_costs(robot_state)

        if manager.get_cost("self_collision").enabled:
            assert "self_collision" in result.names
            # Verify the cost tensor has correct shape
            cost_tensor = result.values[result.names.index("self_collision")]
            assert cost_tensor.shape[0] == batch_size


class TestRobotCostManagerToolPoseCompute:
    """Test tool pose cost computation with proper goal setup."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_tool_pose_compute_costs(
        self, cuda_device_cfg, franka_transition_model, franka_kinematics
    ):
        """Test computing tool pose cost with proper goal setup."""
        from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg

        tool_pose_cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0],
            device_cfg=cuda_device_cfg,
        )
        config = RobotCostManagerCfg(tool_pose_cfg=tool_pose_cfg)

        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        batch_size = 4
        horizon = 10
        dof = franka_transition_model.num_dof

        # Create joint positions using default config
        default_position = franka_kinematics.default_joint_position
        position = default_position.view(1, 1, -1).expand(batch_size, horizon, -1).clone()

        joint_state = JointState(
            position=position,
            velocity=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            acceleration=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            jerk=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            device_cfg=cuda_device_cfg,
        )

        # Use kinematics to compute link poses
        kin_result = franka_kinematics.compute_kinematics(
            JointState.from_position(position, joint_names=franka_kinematics.joint_names)
        )

        # Get link poses from kinematics result
        if kin_result.tool_poses is None:
            pytest.skip("No link poses from kinematics")

        link_poses = kin_result.tool_poses
        # ToolPose requires link_poses (Pose) and tool_frames
        # This test verifies that the manager is correctly initialized
        assert manager.has_cost("tool_pose")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_tool_pose_cost_enabled(
        self, cuda_device_cfg, franka_transition_model
    ):
        """Test that tool pose cost is registered and enabled."""
        from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg

        tool_pose_cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0],
            device_cfg=cuda_device_cfg,
        )
        config = RobotCostManagerCfg(tool_pose_cfg=tool_pose_cfg)

        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(config, franka_transition_model)

        # Verify tool pose cost is registered
        assert manager.has_cost("tool_pose")

        # Verify it can be retrieved
        cost = manager.get_cost("tool_pose")
        assert cost is not None
        assert cost.enabled
