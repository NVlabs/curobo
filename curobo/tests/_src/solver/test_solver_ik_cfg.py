# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for IKSolverCfg configuration class."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


@pytest.fixture(scope="module")
def cuda_device_cfg():
    """Create a CUDA device configuration for testing."""
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


@pytest.fixture(scope="module")
def franka_robot_cfg_dict():
    """Load Franka robot configuration dictionary."""
    return load_yaml(join_path(get_robot_configs_path(), "franka.yml"))


class TestIKSolverCfgDataclassAttributes:
    """Test IKSolverCfg dataclass attributes."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_use_lm_seed_attribute(self, cuda_device_cfg):
        """Test use_lm_seed attribute exists."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "use_lm_seed")
        assert config.use_lm_seed is True  # Default

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_exit_early_attribute(self, cuda_device_cfg):
        """Test exit_early attribute exists."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "exit_early")
        assert config.exit_early is True  # Default

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_exit_early_batch_success_threshold(self, cuda_device_cfg):
        """Test exit_early_batch_success_threshold attribute."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "exit_early_batch_success_threshold")
        assert config.exit_early_batch_success_threshold == 1.0  # Default

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_override_iters_for_multi_link_ik(self, cuda_device_cfg):
        """Test override_iters_for_multi_link_ik attribute."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "override_iters_for_multi_link_ik")


class TestIKSolverCfgCreate:
    """Test IKSolverCfg.create factory method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_basic(self, cuda_device_cfg):
        """Test basic create call."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert isinstance(config, IKSolverCfg)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_with_dict_robot_yaml(self, cuda_device_cfg, franka_robot_cfg_dict):
        """Test create with dict robot config."""
        config = IKSolverCfg.create(
            robot=franka_robot_cfg_dict,
            device_cfg=cuda_device_cfg,
        )
        assert isinstance(config, IKSolverCfg)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_with_robot_cfg_object(self, cuda_device_cfg, franka_robot_cfg_dict):
        """Test create with RobotCfg object."""
        robot_cfg = RobotCfg.create(franka_robot_cfg_dict, cuda_device_cfg)
        config = IKSolverCfg.create(
            robot=robot_cfg,
            device_cfg=cuda_device_cfg,
        )
        assert isinstance(config, IKSolverCfg)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_num_seeds(self, cuda_device_cfg):
        """Test create sets num_seeds."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_seeds=64,
        )
        assert config.num_seeds == 64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_position_tolerance(self, cuda_device_cfg):
        """Test create sets position_tolerance."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            position_tolerance=0.01,
        )
        assert config.position_tolerance == 0.01

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_orientation_tolerance(self, cuda_device_cfg):
        """Test create sets orientation_tolerance."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            orientation_tolerance=0.1,
        )
        assert config.orientation_tolerance == 0.1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_use_cuda_graph(self, cuda_device_cfg):
        """Test create sets use_cuda_graph."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            use_cuda_graph=False,
        )
        assert config.use_cuda_graph is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_random_seed(self, cuda_device_cfg):
        """Test create sets random_seed."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            random_seed=456,
        )
        assert config.random_seed == 456

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_with_scene_model(self, cuda_device_cfg):
        """Test create with scene model."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            scene_model="collision_test.yml",
        )
        assert config.scene_collision_cfg is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_with_self_collision_check_false(self, cuda_device_cfg):
        """Test create with self_collision_check=False."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            self_collision_check=False,
        )
        assert config.self_collision_check is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_with_override_optimizer_num_iters(self, cuda_device_cfg):
        """Test create with override_optimizer_num_iters."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            override_optimizer_num_iters={"particle": 100, "lbfgs": None},
        )
        assert config is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_with_override_iters_for_multi_link_ik(self, cuda_device_cfg):
        """Test create with override_iters_for_multi_link_ik."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            override_iters_for_multi_link_ik=300,
        )
        assert config.override_iters_for_multi_link_ik == 300


class TestIKSolverCfgCreateWithDicts:
    """Test IKSolverCfg.create with dict inputs."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_with_robot_cfg_and_dicts(self, cuda_device_cfg, franka_robot_cfg_dict):
        """Test create with RobotCfg object and dict configs."""
        from curobo.content import get_task_configs_path

        robot_cfg = RobotCfg.create(franka_robot_cfg_dict, cuda_device_cfg)

        # Load config dicts to pass directly
        optimizer_dicts = [
            load_yaml(join_path(get_task_configs_path(), "ik/particle_ik.yml")),
            load_yaml(join_path(get_task_configs_path(), "ik/lbfgs_ik.yml")),
        ]
        metrics_rollout_dict = load_yaml(join_path(get_task_configs_path(), "metrics_base.yml"))
        transition_model_dict = load_yaml(join_path(get_task_configs_path(), "ik/transition_ik.yml"))

        # create() accepts dicts directly - no need for separate from_dict
        config = IKSolverCfg.create(
            robot=robot_cfg,
            optimizer_configs=optimizer_dicts,
            metrics_rollout=metrics_rollout_dict,
            transition_model=transition_model_dict,
            device_cfg=cuda_device_cfg,
        )

        assert isinstance(config, IKSolverCfg)


class TestIKSolverCfgRobotConfig:
    """Test IKSolverCfg robot configuration."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_robot_config(self, cuda_device_cfg):
        """Test config has robot_config."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "robot_config")
        assert config.robot_config is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_device_cfg(self, cuda_device_cfg):
        """Test config has device_cfg."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "device_cfg")
        assert config.device_cfg == cuda_device_cfg


class TestIKSolverCfgOptimizerConfigs:
    """Test IKSolverCfg optimizer configurations."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_optimizer_configs(self, cuda_device_cfg):
        """Test config has optimizer_configs."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "optimizer_configs")
        assert config.optimizer_configs is not None
        assert len(config.optimizer_configs) > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_optimizer_rollout_configs(self, cuda_device_cfg):
        """Test config has optimizer_rollout_configs."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "optimizer_rollout_configs")
        assert config.optimizer_rollout_configs is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_metrics_rollout_config(self, cuda_device_cfg):
        """Test config has metrics_rollout_config."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "metrics_rollout_config")
        assert config.metrics_rollout_config is not None


class TestIKSolverCfgCustomOptimizerYamls:
    """Test IKSolverCfg with custom optimizer yamls."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_with_single_optimizer(self, cuda_device_cfg):
        """Test create with single optimizer."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            optimizer_configs=["ik/lbfgs_ik.yml"],
        )
        assert len(config.optimizer_configs) == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_with_multiple_optimizers(self, cuda_device_cfg):
        """Test create with multiple optimizers."""
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            optimizer_configs=["ik/particle_ik.yml", "ik/lbfgs_ik.yml"],
        )
        assert len(config.optimizer_configs) == 2

