# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for MPCSolverCfg configuration class."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.solver.solver_mpc_cfg import MPCSolverCfg
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


class TestMPCSolverCfgDataclassAttributes:
    """Test MPCSolverCfg dataclass attributes."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_interpolation_steps(self, cuda_device_cfg):
        """Test interpolation_steps attribute exists."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "interpolation_steps")
        assert config.interpolation_steps > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_optimization_dt(self, cuda_device_cfg):
        """Test optimization_dt attribute exists."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "optimization_dt")
        assert config.optimization_dt > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_warm_start_optimization_num_iters(self, cuda_device_cfg):
        """Test warm_start_optimization_num_iters attribute exists."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "warm_start_optimization_num_iters")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_cold_start_optimization_num_iters(self, cuda_device_cfg):
        """Test cold_start_optimization_num_iters attribute exists."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "cold_start_optimization_num_iters")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_use_deceleration_on_failure(self, cuda_device_cfg):
        """Test use_deceleration_on_failure attribute exists."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "use_deceleration_on_failure")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_deceleration_profile(self, cuda_device_cfg):
        """Test deceleration_profile attribute exists."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "deceleration_profile")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_max_deceleration_time(self, cuda_device_cfg):
        """Test max_deceleration_time attribute exists."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "max_deceleration_time")


class TestMPCSolverCfgCreate:
    """Test MPCSolverCfg.create factory method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_basic(self, cuda_device_cfg):
        """Test basic create call."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert isinstance(config, MPCSolverCfg)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_with_dict_robot(self, cuda_device_cfg, franka_robot_cfg_dict):
        """Test create with dict robot config."""
        config = MPCSolverCfg.create(
            robot=franka_robot_cfg_dict,
            device_cfg=cuda_device_cfg,
        )
        assert isinstance(config, MPCSolverCfg)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_with_robot_cfg_object(self, cuda_device_cfg, franka_robot_cfg_dict):
        """Test create with RobotCfg object."""
        robot_cfg = RobotCfg.create(franka_robot_cfg_dict, cuda_device_cfg)
        config = MPCSolverCfg.create(
            robot=robot_cfg,
            device_cfg=cuda_device_cfg,
        )
        assert isinstance(config, MPCSolverCfg)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_optimization_dt(self, cuda_device_cfg):
        """Test create sets optimization_dt."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            optimization_dt=0.05,
        )
        assert config.optimization_dt == 0.05

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_interpolation_steps(self, cuda_device_cfg):
        """Test create sets interpolation_steps."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            interpolation_steps=8,
        )
        assert config.interpolation_steps == 8

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_use_deceleration(self, cuda_device_cfg):
        """Test create sets use_deceleration_on_failure."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            use_deceleration_on_failure=False,
        )
        assert config.use_deceleration_on_failure is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_deceleration_profile(self, cuda_device_cfg):
        """Test create sets deceleration_profile."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            deceleration_profile="linear",
        )
        assert config.deceleration_profile == "linear"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_max_deceleration_time(self, cuda_device_cfg):
        """Test create sets max_deceleration_time."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            max_deceleration_time=3.0,
        )
        assert config.max_deceleration_time == 3.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_use_cuda_graph(self, cuda_device_cfg):
        """Test create sets use_cuda_graph."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            use_cuda_graph=False,
        )
        assert config.use_cuda_graph is False


class TestMPCSolverCfgCreateWithDicts:
    """Test MPCSolverCfg.create with dict inputs."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_with_robot_cfg_and_dicts(self, cuda_device_cfg, franka_robot_cfg_dict):
        """Test create with RobotCfg object and dict configs."""
        from curobo.content import get_task_configs_path

        robot_cfg = RobotCfg.create(franka_robot_cfg_dict, cuda_device_cfg)

        # Load config dicts to pass directly
        optimizer_dicts = [
            load_yaml(join_path(get_task_configs_path(), "mpc/lbfgs_mpc.yml")),
        ]
        metrics_rollout_dict = load_yaml(join_path(get_task_configs_path(), "metrics_base.yml"))
        transition_model_dict = load_yaml(
            join_path(get_task_configs_path(), "mpc/transition_bspline_mpc.yml")
        )

        # create() accepts dicts directly
        config = MPCSolverCfg.create(
            robot=robot_cfg,
            optimizer_configs=optimizer_dicts,
            metrics_rollout=metrics_rollout_dict,
            transition_model=transition_model_dict,
            device_cfg=cuda_device_cfg,
        )

        assert isinstance(config, MPCSolverCfg)


class TestMPCSolverCfgRobotConfig:
    """Test MPCSolverCfg robot configuration."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_robot_config(self, cuda_device_cfg):
        """Test config has robot_config."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.robot_config is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_device_cfg(self, cuda_device_cfg):
        """Test config has device_cfg."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.device_cfg == cuda_device_cfg


class TestMPCSolverCfgOptimizerConfigs:
    """Test MPCSolverCfg optimizer configurations."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_optimizer_configs(self, cuda_device_cfg):
        """Test config has optimizer_configs."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.optimizer_configs is not None
        assert len(config.optimizer_configs) > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_optimizer_rollout_configs(self, cuda_device_cfg):
        """Test config has optimizer_rollout_configs."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.optimizer_rollout_configs is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_metrics_rollout_config(self, cuda_device_cfg):
        """Test config has metrics_rollout_config."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.metrics_rollout_config is not None


class TestMPCSolverCfgCommonFields:
    """Test MPCSolverCfg common solver config fields."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_self_collision_check(self, cuda_device_cfg):
        """Test self_collision_check attribute exists."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            self_collision_check=True,
        )
        assert config.self_collision_check is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_random_seed(self, cuda_device_cfg):
        """Test random_seed attribute exists."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            random_seed=456,
        )
        assert config.random_seed == 456

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_position_tolerance(self, cuda_device_cfg):
        """Test position_tolerance attribute exists."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            position_tolerance=0.01,
        )
        assert config.position_tolerance == 0.01

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_orientation_tolerance(self, cuda_device_cfg):
        """Test orientation_tolerance attribute exists."""
        config = MPCSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            orientation_tolerance=0.1,
        )
        assert config.orientation_tolerance == 0.1

