# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for RobotRolloutCfg.

Tests the rollout_arm_cfg.py module which provides configuration for arm rollouts,
including transition models, cost managers, and scene collision configuration.

Note: This test file focuses on the configuration dataclass structure and validation
logic. It includes tests using real robot configurations (Franka) and task configs
from the curobo/content/configs directory.
"""

# Standard Library

# Third Party
import pytest

# CuRobo
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.rollout.rollout_robot_cfg import RobotRolloutCfg
from curobo._src.transition.robot_state_transition_cfg import (
    RobotStateTransitionCfg,
)
from curobo._src.util_file import (
    get_robot_configs_path,
    get_task_configs_path,
    join_path,
    load_yaml,
)


@pytest.fixture(scope="module")
def franka_robot_cfg():
    """Load Franka robot configuration from YAML file."""
    robot_file = join_path(get_robot_configs_path(), "franka.yml")
    robot_cfg_dict = load_yaml(robot_file)["robot_cfg"]
    # Return the dict, not RobotCfg object, as it's easier for testing
    return robot_cfg_dict


@pytest.fixture(scope="module")
def metrics_base_config():
    """Load base metrics task configuration from YAML file."""
    task_file = join_path(get_task_configs_path(), "metrics_base.yml")
    return load_yaml(task_file)


class TestRobotRolloutCfgInitialization:
    """Test RobotRolloutCfg initialization."""

    def test_default_initialization(self, cuda_device_cfg):
        """Test default configuration initialization."""
        config = RobotRolloutCfg(device_cfg=cuda_device_cfg)

        assert config.device_cfg == cuda_device_cfg
        assert config.sum_horizon is False
        assert config.sampler_seed == 1312
        assert config.cost_manager_config_instance_type == RobotCostManagerCfg
        assert config.transition_model_config_instance_type == RobotStateTransitionCfg
        assert config.transition_model_cfg is None
        assert config.cost_cfg is None
        assert config.constraint_cfg is None
        assert config.hybrid_cost_constraint_cfg is None
        assert config.convergence_cfg is None
        assert config.scene_collision_cfg is None

    def test_initialization_with_custom_parameters(self, cuda_device_cfg):
        """Test initialization with custom parameters."""
        config = RobotRolloutCfg(
            device_cfg=cuda_device_cfg,
            sum_horizon=True,
            sampler_seed=42,
        )

        assert config.device_cfg == cuda_device_cfg
        assert config.sum_horizon is True
        assert config.sampler_seed == 42

    def test_initialization_sets_correct_instance_types(self, cuda_device_cfg):
        """Test that initialization sets correct instance types."""
        config = RobotRolloutCfg(device_cfg=cuda_device_cfg)

        assert config.cost_manager_config_instance_type == RobotCostManagerCfg
        assert config.transition_model_config_instance_type == RobotStateTransitionCfg


class TestRobotRolloutCfgGetCostManagerConfigs:
    """Test RobotRolloutCfg get_cost_manager_configs method."""

    def test_get_cost_manager_configs_empty(self, cuda_device_cfg):
        """Test get_cost_manager_configs with no configs set."""
        config = RobotRolloutCfg(device_cfg=cuda_device_cfg)

        manager_configs = config.get_cost_manager_configs()

        assert isinstance(manager_configs, list)
        assert len(manager_configs) == 0

    def test_get_cost_manager_configs_returns_list(self, cuda_device_cfg):
        """Test that get_cost_manager_configs always returns a list."""
        config = RobotRolloutCfg(device_cfg=cuda_device_cfg)

        manager_configs = config.get_cost_manager_configs()

        assert isinstance(manager_configs, list)


class TestRobotRolloutCfgCreateWithComponentTypes:
    """Test RobotRolloutCfg create_with_component_types class method."""

    def test_create_from_empty_dict(self, cuda_device_cfg):
        """Test creating from empty dictionary with minimal robot cfg."""
        # Create minimal robot config dict
        robot_cfg_dict = {
            "kinematics": {
                "base_link": "base_link",
                "ee_link": "ee_link",
                "cspace": {
                    "joint_names": ["joint1", "joint2", "joint3"],
                    "default_joint_position": [0.0, 0.0, 0.0],
                },
            }
        }

        config = RobotRolloutCfg.create_with_component_types(
            data_dict={},
            robot_cfg=robot_cfg_dict,
            device_cfg=cuda_device_cfg,
        )

        assert config.device_cfg == cuda_device_cfg
        assert config.transition_model_cfg is None
        assert config.cost_cfg is None

    def test_create_with_cost_cfg_dict(self, cuda_device_cfg):
        """Test creating with cost configuration dict."""
        robot_cfg_dict = {
            "kinematics": {
                "base_link": "base_link",
                "ee_link": "ee_link",
                "cspace": {
                    "joint_names": ["joint1", "joint2", "joint3"],
                    "default_joint_position": [0.0, 0.0, 0.0],
                },
            }
        }
        data_dict = {
            "cost_cfg": {
                # Minimal cost manager config
            }
        }

        config = RobotRolloutCfg.create_with_component_types(
            data_dict=data_dict,
            robot_cfg=robot_cfg_dict,
            device_cfg=cuda_device_cfg,
        )

        assert config.cost_cfg is not None
        assert isinstance(config.cost_cfg, RobotCostManagerCfg)

    def test_create_with_constraint_cfg_dict(self, cuda_device_cfg):
        """Test creating with constraint configuration dict."""
        robot_cfg_dict = {
            "kinematics": {
                "base_link": "base_link",
                "ee_link": "ee_link",
                "cspace": {
                    "joint_names": ["joint1", "joint2", "joint3"],
                    "default_joint_position": [0.0, 0.0, 0.0],
                },
            }
        }
        data_dict = {
            "constraint_cfg": {
                # Minimal constraint manager config
            }
        }

        config = RobotRolloutCfg.create_with_component_types(
            data_dict=data_dict,
            robot_cfg=robot_cfg_dict,
            device_cfg=cuda_device_cfg,
        )

        assert config.constraint_cfg is not None
        assert isinstance(config.constraint_cfg, RobotCostManagerCfg)

    def test_create_with_hybrid_cost_constraint_cfg_dict(self, cuda_device_cfg):
        """Test creating with hybrid cost-constraint configuration dict."""
        robot_cfg_dict = {
            "kinematics": {
                "base_link": "base_link",
                "ee_link": "ee_link",
                "cspace": {
                    "joint_names": ["joint1", "joint2", "joint3"],
                    "default_joint_position": [0.0, 0.0, 0.0],
                },
            }
        }
        data_dict = {
            "hybrid_cost_constraint_cfg": {
                # Minimal hybrid config
            }
        }

        config = RobotRolloutCfg.create_with_component_types(
            data_dict=data_dict,
            robot_cfg=robot_cfg_dict,
            device_cfg=cuda_device_cfg,
        )

        assert config.hybrid_cost_constraint_cfg is not None
        assert isinstance(config.hybrid_cost_constraint_cfg, RobotCostManagerCfg)

    def test_create_with_convergence_cfg_dict(self, cuda_device_cfg):
        """Test creating with convergence configuration dict."""
        robot_cfg_dict = {
            "kinematics": {
                "base_link": "base_link",
                "ee_link": "ee_link",
                "cspace": {
                    "joint_names": ["joint1", "joint2", "joint3"],
                    "default_joint_position": [0.0, 0.0, 0.0],
                },
            }
        }
        data_dict = {
            "convergence_cfg": {
                # Minimal convergence config
            }
        }

        config = RobotRolloutCfg.create_with_component_types(
            data_dict=data_dict,
            robot_cfg=robot_cfg_dict,
            device_cfg=cuda_device_cfg,
        )

        assert config.convergence_cfg is not None
        assert isinstance(config.convergence_cfg, RobotCostManagerCfg)

    def test_create_preserves_instance_types(self, cuda_device_cfg):
        """Test that create_with_component_types preserves instance type settings."""
        robot_cfg_dict = {
            "kinematics": {
                "base_link": "base_link",
                "ee_link": "ee_link",
                "cspace": {
                    "joint_names": ["joint1", "joint2", "joint3"],
                    "default_joint_position": [0.0, 0.0, 0.0],
                },
            }
        }
        data_dict = {}

        config = RobotRolloutCfg.create_with_component_types(
            data_dict=data_dict,
            robot_cfg=robot_cfg_dict,
            device_cfg=cuda_device_cfg,
            transition_model_config_instance_type=RobotStateTransitionCfg,
            cost_manager_config_instance_type=RobotCostManagerCfg,
        )

        assert config.transition_model_config_instance_type == RobotStateTransitionCfg
        assert config.cost_manager_config_instance_type == RobotCostManagerCfg

    def test_create_with_none_transition_model_cfg(self, cuda_device_cfg):
        """Test creating with explicitly None transition model config."""
        robot_cfg_dict = {
            "kinematics": {
                "base_link": "base_link",
                "ee_link": "ee_link",
                "cspace": {
                    "joint_names": ["joint1", "joint2", "joint3"],
                    "default_joint_position": [0.0, 0.0, 0.0],
                },
            }
        }
        data_dict = {
            "transition_model_cfg": None,
        }

        config = RobotRolloutCfg.create_with_component_types(
            data_dict=data_dict,
            robot_cfg=robot_cfg_dict,
            device_cfg=cuda_device_cfg,
        )

        assert config.transition_model_cfg is None

    def test_create_with_none_cost_configs(self, cuda_device_cfg):
        """Test creating with explicitly None cost configs."""
        robot_cfg_dict = {
            "kinematics": {
                "base_link": "base_link",
                "ee_link": "ee_link",
                "cspace": {
                    "joint_names": ["joint1", "joint2", "joint3"],
                    "default_joint_position": [0.0, 0.0, 0.0],
                },
            }
        }
        data_dict = {
            "cost_cfg": None,
            "constraint_cfg": None,
            "hybrid_cost_constraint_cfg": None,
            "convergence_cfg": None,
        }

        config = RobotRolloutCfg.create_with_component_types(
            data_dict=data_dict,
            robot_cfg=robot_cfg_dict,
            device_cfg=cuda_device_cfg,
        )

        assert config.cost_cfg is None
        assert config.constraint_cfg is None
        assert config.hybrid_cost_constraint_cfg is None
        assert config.convergence_cfg is None


class TestRobotRolloutCfgEdgeCases:
    """Test RobotRolloutCfg edge cases."""

    def test_config_has_base_fields(self, cuda_device_cfg):
        """Test that RobotRolloutCfg has the standard rollout config fields."""
        config = RobotRolloutCfg(
            device_cfg=cuda_device_cfg,
            sum_horizon=True,
            sampler_seed=99,
        )

        # These fields are directly on RobotRolloutCfg (no inheritance)
        assert hasattr(config, "device_cfg")
        assert hasattr(config, "sum_horizon")
        assert hasattr(config, "sampler_seed")
        assert config.sum_horizon is True
        assert config.sampler_seed == 99

    def test_config_has_arm_specific_attributes(self, cuda_device_cfg):
        """Test that RobotRolloutCfg has arm-specific attributes."""
        config = RobotRolloutCfg(device_cfg=cuda_device_cfg)

        # Arm-specific attributes
        assert hasattr(config, "cost_manager_config_instance_type")
        assert hasattr(config, "transition_model_config_instance_type")
        assert hasattr(config, "transition_model_cfg")
        assert hasattr(config, "cost_cfg")
        assert hasattr(config, "constraint_cfg")
        assert hasattr(config, "hybrid_cost_constraint_cfg")
        assert hasattr(config, "convergence_cfg")
        assert hasattr(config, "scene_collision_cfg")

    def test_config_method_existence(self, cuda_device_cfg):
        """Test that required methods exist."""
        config = RobotRolloutCfg(device_cfg=cuda_device_cfg)

        assert hasattr(config, "get_cost_manager_configs")
        assert callable(config.get_cost_manager_configs)


class TestRobotRolloutCfgWithRealRobotConfig:
    """Test RobotRolloutCfg with real robot configuration (Franka)."""

    def test_create_with_franka_robot(self, franka_robot_cfg, cuda_device_cfg):
        """Test creating configuration with Franka robot."""
        config = RobotRolloutCfg(device_cfg=cuda_device_cfg)

        # Should initialize without errors
        assert config is not None
        assert config.device_cfg == cuda_device_cfg

    def test_create_with_metrics_base_config(
        self, franka_robot_cfg, metrics_base_config, cuda_device_cfg
    ):
        """Test creating configuration using metrics_base.yml task config."""
        # Extract rollout config from metrics_base
        rollout_dict = metrics_base_config.get("rollout", {})

        # Create configuration using the task config structure
        config = RobotRolloutCfg.create_with_component_types(
            data_dict=rollout_dict,
            robot_cfg=franka_robot_cfg,
            device_cfg=cuda_device_cfg,
        )

        assert config is not None
        assert config.device_cfg == cuda_device_cfg

    def test_scene_collision_checker_config_integration(
        self, franka_robot_cfg, metrics_base_config, cuda_device_cfg
    ):
        """Test that scene collision checker config from metrics_base.yml can be used."""
        # metrics_base.yml has scene_collision_checker_cfg
        scene_cfg_dict = metrics_base_config.get("scene_collision_checker_cfg", {})

        # Should contain expected fields
        assert "checker_type" in scene_cfg_dict
        assert scene_cfg_dict["checker_type"] == "PRIMITIVE"



