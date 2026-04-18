# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for RobotStateTransitionCfg and TimeTrajCfg."""

# Third Party
import pytest

# CuRobo
from curobo._src.transition.robot_state_transition_cfg import (
    RobotStateTransitionCfg,
    TimeTrajCfg,
)
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


class TestTimeTrajCfg:
    """Test TimeTrajCfg dataclass."""

    def test_basic_initialization(self):
        """Test basic TimeTrajCfg initialization."""
        cfg = TimeTrajCfg(base_dt=0.02, base_ratio=0.5, max_dt=0.1)

        assert cfg.base_dt == 0.02
        assert cfg.base_ratio == 0.5
        assert cfg.max_dt == 0.1

    def test_get_dt_array_uniform(self):
        """Test get_dt_array with uniform dt (base_ratio=1.0)."""
        cfg = TimeTrajCfg(base_dt=0.02, base_ratio=1.0, max_dt=0.02)
        num_points = 10

        dt_array = cfg.get_dt_array(num_points)

        assert len(dt_array) == num_points
        assert all(dt == pytest.approx(0.02) for dt in dt_array)

    def test_get_dt_array_variable(self):
        """Test get_dt_array with variable dt (base_ratio < 1.0)."""
        cfg = TimeTrajCfg(base_dt=0.02, base_ratio=0.5, max_dt=0.1)
        num_points = 10

        dt_array = cfg.get_dt_array(num_points)

        assert len(dt_array) == num_points
        # First half should be base_dt
        for i in range(5):
            assert dt_array[i] == pytest.approx(0.02)
        # Second half should blend to max_dt
        assert dt_array[-1] == pytest.approx(0.1)

    def test_get_dt_array_single_point(self):
        """Test get_dt_array with single point."""
        cfg = TimeTrajCfg(base_dt=0.02, base_ratio=1.0, max_dt=0.02)

        dt_array = cfg.get_dt_array(1)

        assert len(dt_array) == 1

    def test_update_dt_all(self):
        """Test update_dt with all_dt parameter."""
        cfg = TimeTrajCfg(base_dt=0.02, base_ratio=0.5, max_dt=0.1)

        cfg.update_dt(all_dt=0.05)

        assert cfg.base_dt == 0.05
        assert cfg.max_dt == 0.05

    def test_update_dt_base_dt(self):
        """Test update_dt with base_dt parameter."""
        cfg = TimeTrajCfg(base_dt=0.02, base_ratio=0.5, max_dt=0.1)

        cfg.update_dt(base_dt=0.03)

        assert cfg.base_dt == 0.03
        assert cfg.max_dt == 0.1  # Unchanged

    def test_update_dt_max_dt(self):
        """Test update_dt with max_dt parameter."""
        cfg = TimeTrajCfg(base_dt=0.02, base_ratio=0.5, max_dt=0.1)

        cfg.update_dt(max_dt=0.2)

        assert cfg.base_dt == 0.02  # Unchanged
        assert cfg.max_dt == 0.2

    def test_update_dt_base_ratio(self):
        """Test update_dt with base_ratio parameter."""
        cfg = TimeTrajCfg(base_dt=0.02, base_ratio=0.5, max_dt=0.1)

        cfg.update_dt(base_ratio=0.8)

        assert cfg.base_ratio == 0.8

    def test_update_dt_multiple_params(self):
        """Test update_dt with multiple parameters."""
        cfg = TimeTrajCfg(base_dt=0.02, base_ratio=0.5, max_dt=0.1)

        cfg.update_dt(base_dt=0.03, max_dt=0.2, base_ratio=0.7)

        assert cfg.base_dt == 0.03
        assert cfg.max_dt == 0.2
        assert cfg.base_ratio == 0.7

    def test_update_dt_all_takes_precedence(self):
        """Test that all_dt takes precedence over other parameters."""
        cfg = TimeTrajCfg(base_dt=0.02, base_ratio=0.5, max_dt=0.1)

        cfg.update_dt(all_dt=0.05, base_dt=0.03, max_dt=0.2)

        # all_dt should override base_dt and max_dt
        assert cfg.base_dt == 0.05
        assert cfg.max_dt == 0.05


class TestRobotStateTransitionCfg:
    """Test RobotStateTransitionCfg dataclass."""

    @pytest.fixture(scope="class")
    def franka_robot_cfg_dict(self):
        """Load Franka robot configuration dictionary."""
        robot_file = join_path(get_robot_configs_path(), "franka.yml")
        return load_yaml(robot_file)["robot_cfg"]

    @pytest.fixture(scope="class")
    def device_cfg(self):
        """Create default tensor configuration."""
        return DeviceCfg()

    @pytest.fixture(scope="class")
    def robot_cfg(self, franka_robot_cfg_dict, device_cfg):
        """Create RobotCfg from Franka configuration."""
        return RobotCfg.create({"robot_cfg": franka_robot_cfg_dict}, device_cfg)

    @pytest.fixture(scope="class")
    def time_traj_cfg(self):
        """Create basic TimeTrajCfg."""
        return TimeTrajCfg(base_dt=0.02, base_ratio=0.5, max_dt=0.1)

    def test_basic_initialization(self, robot_cfg, time_traj_cfg, device_cfg):
        """Test basic RobotStateTransitionCfg initialization."""
        cfg = RobotStateTransitionCfg(
            robot_config=robot_cfg,
            dt_traj_params=time_traj_cfg,
            device_cfg=device_cfg,
        )

        assert cfg.robot_config is robot_cfg
        assert cfg.dt_traj_params is time_traj_cfg
        assert cfg.device_cfg is device_cfg
        assert cfg.batch_size == 1
        assert cfg.horizon == 5
        assert cfg.control_space == ControlSpace.ACCELERATION

    def test_initialization_with_custom_batch_size(self, robot_cfg, time_traj_cfg, device_cfg):
        """Test initialization with custom batch size."""
        cfg = RobotStateTransitionCfg(
            robot_config=robot_cfg,
            dt_traj_params=time_traj_cfg,
            device_cfg=device_cfg,
            batch_size=32,
        )

        assert cfg.batch_size == 32

    def test_initialization_with_custom_horizon(self, robot_cfg, time_traj_cfg, device_cfg):
        """Test initialization with custom horizon."""
        cfg = RobotStateTransitionCfg(
            robot_config=robot_cfg,
            dt_traj_params=time_traj_cfg,
            device_cfg=device_cfg,
            horizon=20,
        )

        assert cfg.horizon == 20

    def test_initialization_with_position_control_space(
        self, robot_cfg, time_traj_cfg, device_cfg
    ):
        """Test initialization with POSITION control space."""
        cfg = RobotStateTransitionCfg(
            robot_config=robot_cfg,
            dt_traj_params=time_traj_cfg,
            device_cfg=device_cfg,
            control_space=ControlSpace.POSITION,
        )

        assert cfg.control_space == ControlSpace.POSITION

    def test_initialization_with_teleport_mode(self, robot_cfg, time_traj_cfg, device_cfg):
        """Test initialization with teleport mode."""
        cfg = RobotStateTransitionCfg(
            robot_config=robot_cfg,
            dt_traj_params=time_traj_cfg,
            device_cfg=device_cfg,
            teleport_mode=True,
        )

        assert cfg.teleport_mode is True

    def test_initialization_with_return_full_act_buffer(
        self, robot_cfg, time_traj_cfg, device_cfg
    ):
        """Test initialization with return_full_act_buffer."""
        cfg = RobotStateTransitionCfg(
            robot_config=robot_cfg,
            dt_traj_params=time_traj_cfg,
            device_cfg=device_cfg,
            return_full_act_buffer=True,
        )

        assert cfg.return_full_act_buffer is True

    def test_initialization_with_vel_scale(self, robot_cfg, time_traj_cfg, device_cfg):
        """Test initialization with custom vel_scale."""
        cfg = RobotStateTransitionCfg(
            robot_config=robot_cfg,
            dt_traj_params=time_traj_cfg,
            device_cfg=device_cfg,
            vel_scale=0.5,
        )

        assert cfg.vel_scale == 0.5

    def test_bspline_control_space_requires_n_knots(self, robot_cfg, time_traj_cfg, device_cfg):
        """Test that BSPLINE control space requires n_knots > 5."""
        with pytest.raises(Exception):
            RobotStateTransitionCfg(
                robot_config=robot_cfg,
                dt_traj_params=time_traj_cfg,
                device_cfg=device_cfg,
                control_space=ControlSpace.BSPLINE_4,
                n_knots=4,  # Too few knots
            )

    def test_bspline_control_space_with_valid_n_knots(
        self, robot_cfg, time_traj_cfg, device_cfg
    ):
        """Test BSPLINE control space with valid n_knots."""
        cfg = RobotStateTransitionCfg(
            robot_config=robot_cfg,
            dt_traj_params=time_traj_cfg,
            device_cfg=device_cfg,
            control_space=ControlSpace.BSPLINE_4,
            n_knots=6,
            interpolation_steps=4,
        )

        assert cfg.control_space == ControlSpace.BSPLINE_4
        assert cfg.n_knots == 6
        # Horizon should be computed automatically for bspline
        expected_horizon = ControlSpace.spline_total_interpolation_steps(
            ControlSpace.BSPLINE_4, 6, 4
        )
        assert cfg.horizon == expected_horizon

    def test_bspline_rejects_teleport_mode(self, robot_cfg, time_traj_cfg, device_cfg):
        """Test that BSPLINE control space rejects teleport mode."""
        with pytest.raises(Exception):
            RobotStateTransitionCfg(
                robot_config=robot_cfg,
                dt_traj_params=time_traj_cfg,
                device_cfg=device_cfg,
                control_space=ControlSpace.BSPLINE_4,
                n_knots=6,
                teleport_mode=True,
            )

    def test_bspline_interpolation_steps_bounds(self, robot_cfg, time_traj_cfg, device_cfg):
        """Test that interpolation_steps must be in valid range for bspline."""
        with pytest.raises(Exception):
            RobotStateTransitionCfg(
                robot_config=robot_cfg,
                dt_traj_params=time_traj_cfg,
                device_cfg=device_cfg,
                control_space=ControlSpace.BSPLINE_4,
                n_knots=6,
                interpolation_steps=0,  # Invalid
            )

        with pytest.raises(Exception):
            RobotStateTransitionCfg(
                robot_config=robot_cfg,
                dt_traj_params=time_traj_cfg,
                device_cfg=device_cfg,
                control_space=ControlSpace.BSPLINE_4,
                n_knots=6,
                interpolation_steps=33,  # Too large
            )


class TestRobotStateTransitionCfgCreate:
    """Test RobotStateTransitionCfg.create method."""

    @pytest.fixture(scope="class")
    def franka_robot_cfg_dict(self):
        """Load Franka robot configuration dictionary."""
        robot_file = join_path(get_robot_configs_path(), "franka.yml")
        return load_yaml(robot_file)["robot_cfg"]

    @pytest.fixture(scope="class")
    def device_cfg(self):
        """Create default tensor configuration."""
        return DeviceCfg()

    def test_create_basic(self, franka_robot_cfg_dict, device_cfg):
        """Test create with basic configuration."""
        data_dict = {
            "dt_traj_params": {
                "base_dt": 0.02,
                "base_ratio": 1.0,
                "max_dt": 0.02,
            },
            "control_space": "POSITION",
            "teleport_mode": True,
            "batch_size": 2,
            "horizon": 10,
            "state_filter_cfg": {
                "filter_coeff": {},
                "enable": False,
            },
        }

        cfg = RobotStateTransitionCfg.create(data_dict, franka_robot_cfg_dict, device_cfg)

        assert cfg is not None
        assert cfg.control_space == ControlSpace.POSITION
        assert cfg.teleport_mode is True
        assert cfg.batch_size == 2
        assert cfg.horizon == 10

    def test_create_with_robot_cfg_object(self, franka_robot_cfg_dict, device_cfg):
        """Test create with RobotCfg object instead of dict."""
        robot_cfg = RobotCfg.create({"robot_cfg": franka_robot_cfg_dict}, device_cfg)

        data_dict = {
            "dt_traj_params": {
                "base_dt": 0.02,
                "base_ratio": 1.0,
                "max_dt": 0.02,
            },
            "control_space": "ACCELERATION",
            "teleport_mode": False,
            "batch_size": 4,
            "horizon": 20,
            "state_filter_cfg": {
                "filter_coeff": {},
                "enable": False,
            },
        }

        cfg = RobotStateTransitionCfg.create(data_dict, robot_cfg, device_cfg)

        assert cfg is not None
        assert cfg.control_space == ControlSpace.ACCELERATION
        assert cfg.robot_config is robot_cfg

