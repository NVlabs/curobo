# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for SeedIKSolverCfg configuration class."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.solver.seed_ik.seed_ik_solver_cfg import SeedIKSolverCfg
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


class TestSeedIKSolverCfgDataclassAttributes:
    """Test SeedIKSolverCfg dataclass attributes."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_max_iterations(self, cuda_device_cfg):
        """Test max_iterations attribute exists."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "max_iterations")
        assert config.max_iterations > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_inner_iterations(self, cuda_device_cfg):
        """Test inner_iterations attribute exists."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "inner_iterations")
        assert config.inner_iterations > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_position_tolerance(self, cuda_device_cfg):
        """Test position_tolerance attribute exists."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "position_tolerance")
        assert config.position_tolerance > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_orientation_tolerance(self, cuda_device_cfg):
        """Test orientation_tolerance attribute exists."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "orientation_tolerance")
        assert config.orientation_tolerance > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_lambda_initial(self, cuda_device_cfg):
        """Test lambda_initial attribute exists."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "lambda_initial")
        assert config.lambda_initial > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_lambda_factor(self, cuda_device_cfg):
        """Test lambda_factor attribute exists."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "lambda_factor")
        assert config.lambda_factor > 1.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_lambda_min(self, cuda_device_cfg):
        """Test lambda_min attribute exists."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "lambda_min")
        assert config.lambda_min > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_lambda_max(self, cuda_device_cfg):
        """Test lambda_max attribute exists."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "lambda_max")
        assert config.lambda_max > config.lambda_min

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_num_seeds(self, cuda_device_cfg):
        """Test num_seeds attribute exists."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "num_seeds")
        assert config.num_seeds >= 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_use_cuda_graph(self, cuda_device_cfg):
        """Test use_cuda_graph attribute exists."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert hasattr(config, "use_cuda_graph")


class TestSeedIKSolverCfgFromRobotYaml:
    """Test SeedIKSolverCfg.from_robot_yaml factory method."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_from_robot_yaml_basic(self, cuda_device_cfg):
        """Test basic from_robot_yaml call."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert isinstance(config, SeedIKSolverCfg)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_from_robot_yaml_with_dict(self, cuda_device_cfg, franka_robot_cfg_dict):
        """Test from_robot_yaml with dict robot config."""
        config = SeedIKSolverCfg.create(
            robot=franka_robot_cfg_dict,
            device_cfg=cuda_device_cfg,
        )
        assert isinstance(config, SeedIKSolverCfg)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_from_robot_yaml_with_robot_cfg(self, cuda_device_cfg, franka_robot_cfg_dict):
        """Test from_robot_yaml with RobotCfg object."""
        robot_cfg = RobotCfg.create(franka_robot_cfg_dict, cuda_device_cfg)
        config = SeedIKSolverCfg.create(
            robot=robot_cfg,
            device_cfg=cuda_device_cfg,
        )
        assert isinstance(config, SeedIKSolverCfg)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_from_robot_yaml_custom_max_iterations(self, cuda_device_cfg):
        """Test from_robot_yaml with custom max_iterations."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            max_iterations=32,
            inner_iterations=8,
        )
        assert config.max_iterations == 32
        assert config.inner_iterations == 8

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_from_robot_yaml_custom_tolerances(self, cuda_device_cfg):
        """Test from_robot_yaml with custom tolerances."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            position_tolerance=0.01,
            orientation_tolerance=0.1,
        )
        assert config.position_tolerance == 0.01
        assert config.orientation_tolerance == 0.1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_from_robot_yaml_custom_damping(self, cuda_device_cfg):
        """Test from_robot_yaml with custom damping parameters."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            lambda_initial=0.5,
            lambda_factor=3.0,
        )
        assert config.lambda_initial == 0.5
        assert config.lambda_factor == 3.0


class TestSeedIKSolverCfgRobotConfig:
    """Test SeedIKSolverCfg robot configuration."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_robot_config(self, cuda_device_cfg):
        """Test config has robot_config."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.robot_config is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_has_device_cfg(self, cuda_device_cfg):
        """Test config has device_cfg."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.device_cfg == cuda_device_cfg


class TestSeedIKSolverCfgPostInit:
    """Test SeedIKSolverCfg __post_init__ validation."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_post_init_raises_on_invalid_iterations(self, cuda_device_cfg):
        """Test __post_init__ raises when max_iterations < inner_iterations."""
        with pytest.raises(ValueError):
            SeedIKSolverCfg.create(
                robot="franka.yml",
                device_cfg=cuda_device_cfg,
                max_iterations=4,
                inner_iterations=8,  # inner > max
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_post_init_raises_on_non_divisible_iterations(self, cuda_device_cfg):
        """Test __post_init__ raises when max_iterations not divisible by inner_iterations."""
        with pytest.raises(ValueError):
            SeedIKSolverCfg.create(
                robot="franka.yml",
                device_cfg=cuda_device_cfg,
                max_iterations=15,
                inner_iterations=4,  # 15 % 4 != 0
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_post_init_accepts_valid_iterations(self, cuda_device_cfg):
        """Test __post_init__ accepts valid iteration configuration."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            max_iterations=16,
            inner_iterations=4,  # 16 % 4 == 0
        )
        assert config.max_iterations == 16
        assert config.inner_iterations == 4


class TestSeedIKSolverCfgDefaults:
    """Test SeedIKSolverCfg default values."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_default_max_iterations(self, cuda_device_cfg):
        """Test default max_iterations is 16."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.max_iterations == 16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_default_inner_iterations(self, cuda_device_cfg):
        """Test default inner_iterations is 4."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.inner_iterations == 4

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_default_position_tolerance(self, cuda_device_cfg):
        """Test default position_tolerance is 0.005."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.position_tolerance == 0.005

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_default_orientation_tolerance(self, cuda_device_cfg):
        """Test default orientation_tolerance is 0.05."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.orientation_tolerance == 0.05

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_default_num_seeds(self, cuda_device_cfg):
        """Test default num_seeds is 1."""
        config = SeedIKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
        )
        assert config.num_seeds == 1

