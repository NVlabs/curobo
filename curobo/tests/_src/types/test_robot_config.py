# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for robot configuration types."""

# Standard Library

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.robot.kinematics.kinematics import KinematicsCfg
from curobo._src.robot.types.cspace_params import CSpaceParams
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util_file import get_assets_path, get_robot_configs_path, join_path, load_yaml


def test_cspace_config():
    cspace_config = CSpaceParams(
        ["j0", "j1", "j2", "j3"],
        default_joint_position=[i for i in range(4)],
        cspace_distance_weight=[i for i in range(4)],
        null_space_weight=[i for i in range(4)],
    )

    new_order = ["j3", "j1"]
    cspace_config.inplace_reindex(new_order)
    assert cspace_config.default_joint_position[0] == 3 and cspace_config.default_joint_position[1] == 1
    assert cspace_config.null_space_weight[0] == 3 and cspace_config.null_space_weight[1] == 1
    assert (
        cspace_config.cspace_distance_weight[0] == 3
        and cspace_config.cspace_distance_weight[1] == 1
    )


def test_joint_state():
    device_cfg = DeviceCfg()
    j_names = ["j0", "j1", "j2", "j3"]
    loc_j = ["j4", "jb"]
    final_j = ["jb", "j0", "j1", "j2", "j3", "j4"]

    position = device_cfg.to_device([i for i in range(len(j_names))])

    loc_position = device_cfg.to_device([i + len(j_names) for i in range(len(loc_j))])

    js_1 = JointState.from_position(position, joint_names=j_names)
    js_lock = JointState.from_position(loc_position, loc_j)

    final_js = js_1.get_augmented_joint_state(final_j, js_lock)
    assert final_js.joint_names == final_j
    assert (
        torch.linalg.norm(final_js.position - device_cfg.to_device([5, 0, 1, 2, 3, 4])).item()
        < 1e-8
    )


def test_batch_joint_state():
    device_cfg = DeviceCfg()
    j_names = ["j0", "j1", "j2", "j3"]
    loc_j = ["j4", "jb"]
    final_j = ["jb", "j0", "j1", "j2", "j3", "j4"]

    # $position = device_cfg.to_device([i for i in range(len(j_names))])
    position = torch.zeros((10, len(j_names)), device=device_cfg.device, dtype=device_cfg.dtype)
    for i in range(len(j_names)):
        position[:, i] = i

    loc_position = device_cfg.to_device([i + len(j_names) for i in range(len(loc_j))])

    js_1 = JointState.from_position(position, joint_names=j_names)
    js_lock = JointState.from_position(loc_position, loc_j)

    final_js = js_1.get_augmented_joint_state(final_j, js_lock)
    assert final_js.joint_names == final_j
    assert (
        torch.linalg.norm(
            final_js.position - device_cfg.to_device([5, 0, 1, 2, 3, 4]).unsqueeze(0)
        ).item()
        < 1e-8
    )


class TestRobotCfg:
    """Test RobotCfg class."""

    @pytest.fixture(scope="class")
    def robot_yaml_data(self):
        """Load robot YAML data for testing."""
        return load_yaml(join_path(get_robot_configs_path(), "franka.yml"))

    @pytest.fixture(scope="class")
    def urdf_path(self, robot_yaml_data):
        """Get URDF path for testing."""
        return join_path(
            get_assets_path(), robot_yaml_data["robot_cfg"]["kinematics"]["urdf_path"]
        )

    def test_robot_cfg_initialization(self):
        """Test basic RobotCfg initialization."""
        device_cfg = DeviceCfg()
        kinematics = KinematicsCfg.from_robot_yaml_file("franka.yml", ["panda_hand"])
        robot_cfg = RobotCfg(kinematics=kinematics, device_cfg=device_cfg)

        assert robot_cfg.kinematics is not None
        assert robot_cfg.device_cfg == device_cfg
        assert robot_cfg.dynamics is None

    def test_robot_cfg_create(self, robot_yaml_data):
        """Test RobotCfg.create method."""
        device_cfg = DeviceCfg()
        robot_cfg = RobotCfg.create(robot_yaml_data, device_cfg=device_cfg)

        assert robot_cfg is not None
        assert robot_cfg.kinematics is not None
        assert robot_cfg.device_cfg == device_cfg
        assert robot_cfg.dynamics is None

    def test_robot_cfg_create_with_robot_cfg_key(self, robot_yaml_data):
        """Test RobotCfg.create handles nested 'robot_cfg' key."""
        device_cfg = DeviceCfg()
        # Test with already nested data
        robot_cfg = RobotCfg.create({"robot_cfg": robot_yaml_data["robot_cfg"]}, device_cfg)

        assert robot_cfg is not None
        assert robot_cfg.kinematics is not None

    def test_robot_cfg_create_with_dynamics(self, robot_yaml_data):
        """Test RobotCfg.create with load_dynamics flag."""
        device_cfg = DeviceCfg()
        data = robot_yaml_data.copy()
        data["robot_cfg"]["load_dynamics"] = True

        robot_cfg = RobotCfg.create(data, device_cfg=device_cfg)

        assert robot_cfg is not None
        assert robot_cfg.kinematics is not None
        assert robot_cfg.dynamics is not None

    def test_robot_cfg_from_basic(self, urdf_path):
        """Test RobotCfg.from_basic method."""
        device_cfg = DeviceCfg()
        robot_cfg = RobotCfg.from_basic(
            urdf_path=urdf_path,
            base_link="panda_link0",
            tool_frames=["panda_hand"],
            device_cfg=device_cfg,
            load_dynamics=False,
        )

        assert robot_cfg is not None
        assert robot_cfg.kinematics is not None
        assert robot_cfg.device_cfg == device_cfg
        assert robot_cfg.dynamics is None

    def test_robot_cfg_from_basic_with_dynamics(self, urdf_path):
        """Test RobotCfg.from_basic with load_dynamics=True."""
        device_cfg = DeviceCfg()
        robot_cfg = RobotCfg.from_basic(
            urdf_path=urdf_path,
            base_link="panda_link0",
            tool_frames=["panda_hand"],
            device_cfg=device_cfg,
            load_dynamics=True,
        )

        assert robot_cfg is not None
        assert robot_cfg.kinematics is not None
        assert robot_cfg.dynamics is not None

    def test_robot_cfg_cspace_property(self, robot_yaml_data):
        """Test RobotCfg.cspace property."""
        device_cfg = DeviceCfg()
        robot_cfg = RobotCfg.create(robot_yaml_data, device_cfg=device_cfg)

        cspace = robot_cfg.cspace

        assert cspace is not None
        assert isinstance(cspace, CSpaceParams)
        assert len(cspace.joint_names) > 0
