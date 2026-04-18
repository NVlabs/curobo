# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for solver configuration fields (via IKSolverCfg)."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.solver.solver_core_cfg import create_scene_collision_cfg
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util_file import join_path, load_yaml


@pytest.fixture(scope="module")
def cuda_device_cfg():
    """Create a CUDA device configuration for testing."""
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


class TestSolverCfgDataclass:
    """Test solver config dataclass attributes."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_dataclass_default_values(self, cuda_device_cfg):
        """Test default values through IKSolverCfg (concrete implementation)."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )

        assert config.num_seeds == 32  # Default
        assert config.position_tolerance == 0.005  # Default
        assert config.orientation_tolerance == 0.05  # Default
        assert config.random_seed == 123  # Default
        assert config.self_collision_check is True  # Default

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_robot_config(self, cuda_device_cfg):
        """Test config has robot_config."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.robot_config is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_optimizer_configs(self, cuda_device_cfg):
        """Test config has optimizer_configs."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.optimizer_configs is not None
        assert len(config.optimizer_configs) > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_optimizer_rollout_configs(self, cuda_device_cfg):
        """Test config has optimizer_rollout_configs."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.optimizer_rollout_configs is not None
        assert len(config.optimizer_rollout_configs) > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_metrics_rollout_config(self, cuda_device_cfg):
        """Test config has metrics_rollout_config."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.metrics_rollout_config is not None


class TestSolverCfgPostInit:
    """Test solver config __post_init__ behavior."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_store_debug_disables_cuda_graph(self, cuda_device_cfg):
        """Test that store_debug=True disables CUDA graph."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            use_cuda_graph=True,
            store_debug=True,
        )
        # store_debug=True should disable CUDA graph
        assert config.use_cuda_graph is False


class TestSolverCfgCreateSceneCollisionCfg:
    """Test create_scene_collision_cfg function."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_scene_collision_cfg_none(self, cuda_device_cfg):
        """Test creating scene collision cfg returns None when no scene dict."""
        result = create_scene_collision_cfg(
            scene_model_dict=None,
            collision_cache=None,
            device_cfg=cuda_device_cfg,
        )
        assert result is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_scene_collision_cfg_with_scene(self, cuda_device_cfg):
        """Test creating scene collision cfg with scene dict."""
        from curobo._src.util_file import get_scene_configs_path

        scene_model_dict = load_yaml(
            join_path(get_scene_configs_path(), "collision_test.yml")
        )

        result = create_scene_collision_cfg(
            scene_model_dict=scene_model_dict,
            collision_cache=None,
            device_cfg=cuda_device_cfg,
        )
        assert result is not None


class TestSolverCfgTolerances:
    """Test solver config tolerance settings."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_custom_position_tolerance(self, cuda_device_cfg):
        """Test setting custom position tolerance."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            position_tolerance=0.01,
        )
        assert config.position_tolerance == 0.01

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_custom_orientation_tolerance(self, cuda_device_cfg):
        """Test setting custom orientation tolerance."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            orientation_tolerance=0.1,
        )
        assert config.orientation_tolerance == 0.1


class TestSolverCfgNumSeeds:
    """Test solver config num_seeds setting."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_custom_num_seeds(self, cuda_device_cfg):
        """Test setting custom num_seeds."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_seeds=64,
        )
        assert config.num_seeds == 64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_small_num_seeds(self, cuda_device_cfg):
        """Test with small num_seeds."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_seeds=4,
        )
        assert config.num_seeds == 4


class TestSolverCfgCUDAGraph:
    """Test solver config CUDA graph settings."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_graph_enabled(self, cuda_device_cfg):
        """Test CUDA graph can be enabled."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            use_cuda_graph=True,
        )
        assert config.use_cuda_graph is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_graph_disabled(self, cuda_device_cfg):
        """Test CUDA graph can be disabled."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            use_cuda_graph=False,
        )
        assert config.use_cuda_graph is False


class TestSolverCfgRandomSeed:
    """Test solver config random seed settings."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_custom_random_seed(self, cuda_device_cfg):
        """Test setting custom random seed."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            random_seed=456,
        )
        assert config.random_seed == 456


class TestSolverCfgCollisionActivationDistance:
    """Test solver config collision activation distance."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_custom_collision_activation_distance(self, cuda_device_cfg):
        """Test setting custom collision activation distance."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            optimizer_collision_activation_distance=0.02,
        )
        assert config.optimizer_collision_activation_distance == 0.02


class TestSolverCfgSelfCollision:
    """Test solver config self-collision settings."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_self_collision_enabled(self, cuda_device_cfg):
        """Test self-collision check enabled."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            self_collision_check=True,
        )
        assert config.self_collision_check is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_self_collision_disabled(self, cuda_device_cfg):
        """Test self-collision check disabled."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            self_collision_check=False,
        )
        assert config.self_collision_check is False

