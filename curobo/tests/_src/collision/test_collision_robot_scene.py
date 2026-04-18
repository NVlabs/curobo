# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for RobotSceneCollision and RobotSceneCollisionCfg.

Tests the robot-scene collision checking functionality including:
- Configuration loading
- Collision distance computation
- Self-collision detection
- Joint bound validation
- Sampling collision-free configurations
"""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.collision.collision_robot_scene import RobotSceneCollision
from curobo._src.collision.collision_robot_scene_cfg import RobotSceneCollisionCfg


class TestRobotSceneCollisionCfg:
    """Tests for RobotSceneCollisionCfg configuration class."""

    def test_import(self):
        """Verify module imports correctly."""
        assert RobotSceneCollisionCfg is not None

    def test_load_from_config_robot_only(self, cuda_device_cfg):
        """Test loading config with robot only (no scene)."""
        cfg = RobotSceneCollisionCfg.load_from_config(
            robot_config="franka.yml",
            scene_model="collision_table.yml",
            device_cfg=cuda_device_cfg,
        )
        assert cfg is not None
        assert cfg.kinematics is not None
        assert cfg.sampler is not None
        assert cfg.cspace_cost is not None
        assert cfg.self_collision_cost is not None



class TestRobotSceneCollision:
    """Tests for RobotSceneCollision class."""

    @pytest.fixture(scope="class")
    def collision_checker(self, cuda_device_cfg):
        """Create a collision checker for testing."""
        cfg = RobotSceneCollisionCfg.load_from_config(
            robot_config="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        collision_checker_instance = RobotSceneCollision(cfg)
        collision_checker_instance.setup_batch_tensors(5, 1)
        return collision_checker_instance

    def test_import(self):
        """Verify module imports correctly."""
        assert RobotSceneCollision is not None

    def test_instantiation(self, cuda_device_cfg):
        """Test basic instantiation."""
        cfg = RobotSceneCollisionCfg.load_from_config(
            robot_config="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        checker = RobotSceneCollision(cfg)
        assert checker is not None

    def test_get_kinematics(self, collision_checker):
        """Test forward kinematics computation."""
        batch_size = 5
        dof = collision_checker.kinematics.get_dof()
        q = torch.zeros((batch_size, 1, dof), device=collision_checker.device_cfg.device)
        state = collision_checker.get_kinematics(q)
        assert state is not None
        assert state.robot_spheres is not None
        assert state.robot_spheres.shape[0] == batch_size

    def test_get_kinematics_1d_raises(self, collision_checker):
        """Test that 1D input raises error."""
        dof = collision_checker.kinematics.get_dof()
        q = torch.zeros(dof, device=collision_checker.device_cfg.device)
        with pytest.raises(Exception):
            collision_checker.get_kinematics(q)

    def test_get_self_collision_distance(self, collision_checker):
        """Test self-collision distance computation."""
        batch_size = 5
        horizon = 1
        dof = collision_checker.kinematics.get_dof()
        q = torch.zeros((batch_size, horizon, dof), device=collision_checker.device_cfg.device)
        q_default  = collision_checker.kinematics.default_joint_position
        q[:, 0, :] = q_default
        state = collision_checker.get_kinematics(q)
        spheres = state.robot_spheres
        d_self = collision_checker.get_self_collision_distance(spheres)
        assert d_self is not None
        assert d_self.shape[0] == batch_size

    def test_get_bound(self, collision_checker):
        """Test joint bound violation computation."""
        batch_size = 5
        dof = collision_checker.kinematics.get_dof()
        q = torch.zeros((batch_size, 1, dof), device=collision_checker.device_cfg.device)
        q_default  = collision_checker.kinematics.default_joint_position
        q[:, 0, :] = q_default
        d_bound = collision_checker.get_bound(q)
        assert d_bound is not None
        # Default joint position should be within bounds
        assert torch.all(d_bound == 0.0)

    def test_validate(self, collision_checker):
        """Test configuration validation."""
        batch_size = 10
        dof = collision_checker.kinematics.get_dof()
        q = torch.zeros((batch_size, 1, dof), device=collision_checker.device_cfg.device)
        mask = collision_checker.validate(q)
        assert mask is not None
        assert mask.shape[0] == batch_size
        assert mask.dtype == torch.bool

    def test_sample(self, collision_checker):
        """Test collision-free sampling."""
        n_samples = 5
        q = collision_checker.sample(n_samples, mask_valid=False)
        assert q is not None
        assert q.shape[0] >= 1  # May have fewer if rejection sampling

    def test_link_names_property(self, collision_checker):
        """Test tool_frames property."""
        tool_frames = collision_checker.tool_frames
        assert tool_frames is not None
        assert len(tool_frames) > 0


