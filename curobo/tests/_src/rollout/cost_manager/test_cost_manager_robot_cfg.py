# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RobotCostManagerCfg.

Tests the flat cost manager configuration dataclass that holds every cost,
constraint, and convergence field used by the robot rollout.
"""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.cost.cost_cspace_cfg import CSpaceCostCfg
from curobo._src.cost.cost_cspace_dist_cfg import CSpaceDistCostCfg
from curobo._src.cost.cost_cspace_type import CSpaceCostType
from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg


class TestRobotCostManagerCfgInitialization:
    """Test RobotCostManagerCfg initialization."""

    def test_default_initialization(self):
        """Test default initialization with all fields None."""
        cfg = RobotCostManagerCfg()

        assert cfg.class_type is RobotCostManager
        assert cfg.self_collision_cfg is None
        assert cfg.scene_collision_cfg is None
        assert cfg.cspace_cfg is None
        assert cfg.start_cspace_dist_cfg is None
        assert cfg.target_cspace_dist_cfg is None
        assert cfg.tool_pose_cfg is None

    def test_initialization_with_start_cspace_dist_cfg(self, cuda_device_cfg):
        start_cspace_dist_cfg = CSpaceDistCostCfg(weight=1.0, device_cfg=cuda_device_cfg)
        cfg = RobotCostManagerCfg(start_cspace_dist_cfg=start_cspace_dist_cfg)

        assert cfg.start_cspace_dist_cfg is start_cspace_dist_cfg

    def test_initialization_with_target_cspace_dist_cfg(self, cuda_device_cfg):
        target_cspace_dist_cfg = CSpaceDistCostCfg(weight=0.5, device_cfg=cuda_device_cfg)
        cfg = RobotCostManagerCfg(target_cspace_dist_cfg=target_cspace_dist_cfg)

        assert cfg.target_cspace_dist_cfg is target_cspace_dist_cfg

    def test_initialization_with_tool_pose_cfg(self, cuda_device_cfg):
        tool_pose_cfg = ToolPoseCostCfg(weight=[1.0, 1.0], device_cfg=cuda_device_cfg)
        cfg = RobotCostManagerCfg(tool_pose_cfg=tool_pose_cfg)

        assert cfg.tool_pose_cfg is tool_pose_cfg


class TestRobotCostManagerCfgDataclass:
    """Test dataclass behavior."""

    def test_is_dataclass(self):
        from dataclasses import is_dataclass

        assert is_dataclass(RobotCostManagerCfg)

    def test_dataclass_fields(self):
        from dataclasses import fields

        field_names = [f.name for f in fields(RobotCostManagerCfg)]

        assert "class_type" in field_names
        assert "self_collision_cfg" in field_names
        assert "scene_collision_cfg" in field_names
        assert "cspace_cfg" in field_names
        assert "start_cspace_dist_cfg" in field_names
        assert "target_cspace_dist_cfg" in field_names
        assert "tool_pose_cfg" in field_names

    def test_no_inheritance(self):
        """RobotCostManagerCfg should not inherit from any cost manager config."""
        assert RobotCostManagerCfg.__bases__ == (object,)


class TestRobotCostManagerCfgCreate:
    """Test create static method."""

    def test_create_empty(self):
        cfg = RobotCostManagerCfg.create({})

        assert isinstance(cfg, RobotCostManagerCfg)
        assert cfg.class_type is RobotCostManager
        assert cfg.self_collision_cfg is None
        assert cfg.scene_collision_cfg is None
        assert cfg.cspace_cfg is None
        assert cfg.start_cspace_dist_cfg is None
        assert cfg.target_cspace_dist_cfg is None
        assert cfg.tool_pose_cfg is None

    def test_create_with_start_cspace_dist_cfg(self, cuda_device_cfg):
        data_dict = {"start_cspace_dist_cfg": {"weight": 2.0}}
        cfg = RobotCostManagerCfg.create(data_dict, device_cfg=cuda_device_cfg)

        assert cfg.start_cspace_dist_cfg is not None
        assert isinstance(cfg.start_cspace_dist_cfg, CSpaceDistCostCfg)

    def test_create_with_target_cspace_dist_cfg(self, cuda_device_cfg):
        data_dict = {"target_cspace_dist_cfg": {"weight": 0.5}}
        cfg = RobotCostManagerCfg.create(data_dict, device_cfg=cuda_device_cfg)

        assert cfg.target_cspace_dist_cfg is not None
        assert isinstance(cfg.target_cspace_dist_cfg, CSpaceDistCostCfg)

    def test_create_with_tool_pose_cfg(self, cuda_device_cfg):
        data_dict = {"tool_pose_cfg": {"weight": [1.0, 1.0]}}
        cfg = RobotCostManagerCfg.create(data_dict, device_cfg=cuda_device_cfg)

        assert cfg.tool_pose_cfg is not None
        assert isinstance(cfg.tool_pose_cfg, ToolPoseCostCfg)

    def test_create_with_cspace_cfg(self, cuda_device_cfg):
        data_dict = {
            "cspace_cfg": {
                "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
                "activation_distance": [0.0, 0.0, 0.0, 0.0, 0.0],
                "cost_type": CSpaceCostType.STATE,
            }
        }
        cfg = RobotCostManagerCfg.create(data_dict, device_cfg=cuda_device_cfg)

        assert cfg.cspace_cfg is not None
        assert isinstance(cfg.cspace_cfg, CSpaceCostCfg)

    def test_create_with_mixed_configs(self, cuda_device_cfg):
        data_dict = {
            "cspace_cfg": {
                "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
                "activation_distance": [0.0, 0.0, 0.0, 0.0, 0.0],
                "cost_type": CSpaceCostType.STATE,
            },
            "start_cspace_dist_cfg": {"weight": 2.0},
            "target_cspace_dist_cfg": {"weight": 0.5},
            "tool_pose_cfg": {"weight": [1.0, 1.0]},
        }
        cfg = RobotCostManagerCfg.create(data_dict, device_cfg=cuda_device_cfg)

        assert cfg.cspace_cfg is not None
        assert cfg.start_cspace_dist_cfg is not None
        assert cfg.target_cspace_dist_cfg is not None
        assert cfg.tool_pose_cfg is not None

    def test_create_ignores_unknown_keys(self, cuda_device_cfg):
        data_dict = {
            "unknown_key": {"weight": 1.0},
            "start_cspace_dist_cfg": {"weight": 2.0},
        }
        cfg = RobotCostManagerCfg.create(data_dict, device_cfg=cuda_device_cfg)

        assert cfg.start_cspace_dist_cfg is not None
        assert not hasattr(cfg, "unknown_key")


class TestRobotCostManagerCfgCreateSceneCollision:
    """Test create with scene collision configuration."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_with_scene_collision_checker(self, cuda_device_cfg):
        from curobo._src.geom.collision import SceneCollisionCfg, create_collision_checker

        world_cfg = SceneCollisionCfg(
            device_cfg=cuda_device_cfg,
            max_distance=0.1,
            cache={"primitive": 10},
        )
        scene_collision_checker = create_collision_checker(world_cfg)

        data_dict = {"scene_collision_cfg": {"weight": 1.0}}
        cfg = RobotCostManagerCfg.create(
            data_dict,
            device_cfg=cuda_device_cfg,
            scene_collision_checker=scene_collision_checker,
        )

        assert cfg.scene_collision_cfg is not None
        assert cfg.scene_collision_cfg.scene_collision_checker is scene_collision_checker

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_scene_collision_without_checker(self, cuda_device_cfg):
        data_dict = {"scene_collision_cfg": {"weight": 1.0}}
        cfg = RobotCostManagerCfg.create(data_dict, device_cfg=cuda_device_cfg)

        assert cfg.scene_collision_cfg is not None
        assert cfg.scene_collision_cfg._scene_collision_checker is None


class TestRobotCostManagerCfgUtilityMethods:
    """Test utility methods exposed by RobotCostManagerCfg."""

    def test_has_update_collision_activation_distance(self):
        cfg = RobotCostManagerCfg()
        assert hasattr(cfg, "update_collision_activation_distance")

    def test_has_disable_self_collision(self):
        cfg = RobotCostManagerCfg()
        assert hasattr(cfg, "disable_self_collision")

    def test_has_update_regularization_weight(self):
        cfg = RobotCostManagerCfg()
        assert hasattr(cfg, "update_regularization_weight")

    def test_disable_self_collision_noop_when_none(self):
        cfg = RobotCostManagerCfg()
        cfg.disable_self_collision()  # should not raise

    def test_update_collision_activation_distance_noop_when_none(self):
        cfg = RobotCostManagerCfg()
        cfg.update_collision_activation_distance(0.05)  # should not raise
