# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for TrajOptSolverCfg configuration class."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.trajectory import TrajInterpolationType
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


class TestTrajOptSolverCfgDataclassAttributes:
    """Test TrajOptSolverCfg dataclass attributes."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_minimum_trajectory_dt(self, cuda_device_cfg):
        """Test minimum_trajectory_dt attribute exists."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "minimum_trajectory_dt")
        assert config.minimum_trajectory_dt > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_maximum_trajectory_dt(self, cuda_device_cfg):
        """Test maximum_trajectory_dt attribute exists."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "maximum_trajectory_dt")
        assert config.maximum_trajectory_dt > config.minimum_trajectory_dt

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_interpolation_dt(self, cuda_device_cfg):
        """Test interpolation_dt attribute exists."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "interpolation_dt")
        assert config.interpolation_dt > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_interpolation_type(self, cuda_device_cfg):
        """Test interpolation_type attribute exists."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "interpolation_type")
        assert isinstance(config.interpolation_type, TrajInterpolationType)


class TestTrajOptSolverCfgCreate:
    """Test TrajOptSolverCfg.create factory method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_basic(self, cuda_device_cfg):
        """Test basic create call."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert isinstance(config, TrajOptSolverCfg)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_with_dict_robot(self, cuda_device_cfg, franka_robot_cfg_dict):
        """Test create with dict robot config."""
        config = TrajOptSolverCfg.create(
            robot=franka_robot_cfg_dict,
            device_cfg=cuda_device_cfg,
        )
        assert isinstance(config, TrajOptSolverCfg)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_with_robot_cfg_object(self, cuda_device_cfg, franka_robot_cfg_dict):
        """Test create with RobotCfg object."""
        robot_cfg = RobotCfg.create(franka_robot_cfg_dict, cuda_device_cfg)
        config = TrajOptSolverCfg.create(
            robot=robot_cfg,
            device_cfg=cuda_device_cfg,
        )
        assert isinstance(config, TrajOptSolverCfg)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_num_seeds(self, cuda_device_cfg):
        """Test create sets num_seeds."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_seeds=8,
        )
        assert config.num_seeds == 8

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_default_num_seeds(self, cuda_device_cfg):
        """Test default num_seeds is 4 for trajopt."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.num_seeds == 4  # TrajOpt default

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_position_tolerance(self, cuda_device_cfg):
        """Test create sets position_tolerance."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            position_tolerance=0.01,
        )
        assert config.position_tolerance == 0.01

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_orientation_tolerance(self, cuda_device_cfg):
        """Test create sets orientation_tolerance."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            orientation_tolerance=0.1,
        )
        assert config.orientation_tolerance == 0.1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_minimum_trajectory_dt(self, cuda_device_cfg):
        """Test create sets minimum_trajectory_dt."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            minimum_trajectory_dt=0.005,
        )
        assert config.minimum_trajectory_dt == 0.005

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_maximum_trajectory_dt(self, cuda_device_cfg):
        """Test create sets maximum_trajectory_dt."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            maximum_trajectory_dt=0.1,
        )
        assert config.maximum_trajectory_dt == 0.1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_use_cuda_graph(self, cuda_device_cfg):
        """Test create sets use_cuda_graph."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            use_cuda_graph=False,
        )
        assert config.use_cuda_graph is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_with_scene_model(self, cuda_device_cfg):
        """Test create with scene model."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            scene_model="collision_test.yml",
        )
        assert config.scene_collision_cfg is not None


class TestTrajOptSolverCfgCreateWithDicts:
    """Test TrajOptSolverCfg.create with dict inputs."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_with_robot_cfg_and_dicts(self, cuda_device_cfg, franka_robot_cfg_dict):
        """Test create with RobotCfg object and dict configs."""
        from curobo.content import get_task_configs_path

        robot_cfg = RobotCfg.create(franka_robot_cfg_dict, cuda_device_cfg)

        # Load config dicts to pass directly
        optimizer_dicts = [
            load_yaml(join_path(get_task_configs_path(), "trajopt/lbfgs_bspline_trajopt.yml")),
        ]
        metrics_rollout_dict = load_yaml(join_path(get_task_configs_path(), "metrics_base.yml"))
        transition_model_dict = load_yaml(
            join_path(get_task_configs_path(), "trajopt/transition_bspline_trajopt.yml")
        )

        # create() accepts dicts directly
        config = TrajOptSolverCfg.create(
            robot=robot_cfg,
            optimizer_configs=optimizer_dicts,
            metrics_rollout=metrics_rollout_dict,
            transition_model=transition_model_dict,
            device_cfg=cuda_device_cfg,
        )

        assert isinstance(config, TrajOptSolverCfg)


class TestTrajOptSolverCfgRobotConfig:
    """Test TrajOptSolverCfg robot configuration."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_robot_config(self, cuda_device_cfg):
        """Test config has robot_config."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.robot_config is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_device_cfg(self, cuda_device_cfg):
        """Test config has device_cfg."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.device_cfg == cuda_device_cfg


class TestTrajOptSolverCfgOptimizerConfigs:
    """Test TrajOptSolverCfg optimizer configurations."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_optimizer_configs(self, cuda_device_cfg):
        """Test config has optimizer_configs."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.optimizer_configs is not None
        assert len(config.optimizer_configs) > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_optimizer_rollout_configs(self, cuda_device_cfg):
        """Test config has optimizer_rollout_configs."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.optimizer_rollout_configs is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_metrics_rollout_config(self, cuda_device_cfg):
        """Test config has metrics_rollout_config."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.metrics_rollout_config is not None


class TestTrajOptSolverCfgInterpolationType:
    """Test TrajOptSolverCfg interpolation type settings."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_default_interpolation_type_bspline(self, cuda_device_cfg):
        """Test default interpolation type is BSPLINE_KNOTS_CUDA."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        # Default transition model uses BSPLINE control space
        assert config.interpolation_type == TrajInterpolationType.BSPLINE_KNOTS_CUDA


class TestTrajOptSolverCfgCommonFields:
    """Test TrajOptSolverCfg common solver config fields."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_self_collision_check(self, cuda_device_cfg):
        """Test self_collision_check attribute exists."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            self_collision_check=True,
        )
        assert config.self_collision_check is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_random_seed(self, cuda_device_cfg):
        """Test random_seed attribute exists."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            random_seed=456,
        )
        assert config.random_seed == 456

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_optimizer_collision_activation_distance(self, cuda_device_cfg):
        """Test optimizer_collision_activation_distance attribute exists."""
        config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            optimizer_collision_activation_distance=0.02,
        )
        assert config.optimizer_collision_activation_distance == 0.02

