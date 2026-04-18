# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for SelfCollisionCost and SelfCollisionCostCfg using real Franka kinematics."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.warp import init_warp
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


@pytest.fixture(scope="module")
def setup_warp():
    """Initialize warp before running any tests in this module."""
    device_cfg = DeviceCfg(device=torch.device("cuda:0"))
    init_warp(quiet=True, device_cfg=device_cfg)
    return device_cfg


@pytest.fixture(scope="module")
def device_cfg(setup_warp):
    """Create tensor configuration for GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return setup_warp


@pytest.fixture(scope="module")
def franka_robot_cfg(device_cfg):
    """Load Franka robot configuration."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    robot_cfg = RobotCfg.create(robot_data["robot_cfg"], device_cfg)
    return robot_cfg


@pytest.fixture(scope="module")
def franka_kinematics(franka_robot_cfg):
    """Create Franka kinematics model."""
    return Kinematics(franka_robot_cfg.kinematics)


@pytest.fixture(scope="module")
def self_collision_kin_config(franka_kinematics):
    """Get self-collision configuration from Franka kinematics model."""
    # Kinematics class has get_self_collision_config() method
    return franka_kinematics.get_self_collision_config()


class TestSelfCollisionCostCfg:
    """Test SelfCollisionCostCfg class."""

    def test_default_init(self, device_cfg, self_collision_kin_config):
        """Test default initialization."""
        cfg = SelfCollisionCostCfg(
            weight=1.0,
            device_cfg=device_cfg,
            self_collision_kin_config=self_collision_kin_config,
        )
        assert cfg.class_type == SelfCollisionCost
        assert cfg.self_collision_kin_config is not None
        assert cfg.store_pair_distance is False

    def test_init_with_store_pair_distance(self, device_cfg, self_collision_kin_config):
        """Test initialization with store_pair_distance=True."""
        cfg = SelfCollisionCostCfg(
            weight=1.0,
            device_cfg=device_cfg,
            self_collision_kin_config=self_collision_kin_config,
            store_pair_distance=True,
        )
        assert cfg.store_pair_distance is True


class TestSelfCollisionCost:
    """Test SelfCollisionCost class."""

    def test_init(self, device_cfg, self_collision_kin_config):
        """Test SelfCollisionCost initialization."""
        cfg = SelfCollisionCostCfg(
            weight=1.0,
            device_cfg=device_cfg,
            self_collision_kin_config=self_collision_kin_config,
        )
        cost = SelfCollisionCost(cfg)
        assert cost is not None
        assert cost.config == cfg

    def test_init_missing_self_collision_kin_config_raises_error(self, device_cfg):
        """Test that missing self_collision_kin_config raises error."""
        cfg = SelfCollisionCostCfg(
            weight=1.0,
            device_cfg=device_cfg,
            self_collision_kin_config=None,
        )
        with pytest.raises(Exception):
            SelfCollisionCost(cfg)

    def test_setup_batch_tensors(self, device_cfg, self_collision_kin_config):
        """Test setup_batch_tensors."""
        cfg = SelfCollisionCostCfg(
            weight=1.0,
            device_cfg=device_cfg,
            self_collision_kin_config=self_collision_kin_config,
        )
        cost = SelfCollisionCost(cfg)
        batch_size = 4
        horizon = 10
        cost.setup_batch_tensors(batch_size, horizon)

        num_spheres = self_collision_kin_config.num_spheres
        assert cost._batch_size == batch_size
        assert cost._horizon == horizon
        assert cost._out_distance.shape == (batch_size, horizon, 1)
        assert cost._out_grad.shape == (batch_size, horizon, num_spheres, 4)
        assert cost._sparse_sphere_idx.shape == (batch_size, horizon, num_spheres)

    def test_setup_batch_tensors_with_store_pair_distance(
        self, device_cfg, self_collision_kin_config
    ):
        """Test setup_batch_tensors with store_pair_distance=True."""
        cfg = SelfCollisionCostCfg(
            weight=1.0,
            device_cfg=device_cfg,
            self_collision_kin_config=self_collision_kin_config,
            store_pair_distance=True,
        )
        cost = SelfCollisionCost(cfg)
        batch_size = 4
        horizon = 10
        cost.setup_batch_tensors(batch_size, horizon)

        num_collision_pairs = self_collision_kin_config.collision_pairs.shape[0]
        assert cost._pair_distance.shape == (batch_size, horizon, num_collision_pairs)

    def test_forward_with_kinematics(
        self, device_cfg, self_collision_kin_config, franka_kinematics
    ):
        """Test forward pass using real Franka kinematics."""
        cfg = SelfCollisionCostCfg(
            weight=1.0,
            device_cfg=device_cfg,
            self_collision_kin_config=self_collision_kin_config,
        )
        cost = SelfCollisionCost(cfg)
        batch_size = 4
        horizon = 10
        dof = franka_kinematics.dof
        cost.setup_batch_tensors(batch_size, horizon)

        # Create random joint configurations
        q = torch.zeros((batch_size, horizon, dof), **device_cfg.as_torch_dict())

        kin_state = franka_kinematics.compute_kinematics(
            JointState.from_position(q, joint_names=franka_kinematics.joint_names)
        )
        robot_spheres = kin_state.robot_spheres

        result = cost.forward(robot_spheres)
        assert result.shape == (batch_size, horizon, 1)

    def test_forward_default_joint_position_no_collision(
        self, device_cfg, self_collision_kin_config, franka_kinematics, franka_robot_cfg
    ):
        """Test forward with default joint configuration (should have no self-collision)."""
        cfg = SelfCollisionCostCfg(
            weight=1.0,
            device_cfg=device_cfg,
            self_collision_kin_config=self_collision_kin_config,
        )
        cost = SelfCollisionCost(cfg)
        batch_size = 1
        horizon = 1
        dof = franka_kinematics.dof
        cost.setup_batch_tensors(batch_size, horizon)

        # Use default joint configuration (should be collision-free)
        # default_joint_position is in the cspace attribute
        default_position = franka_robot_cfg.kinematics.kinematics_config.cspace.default_joint_position
        q = default_position.view(1, 1, dof)

        kin_state = franka_kinematics.compute_kinematics(
            JointState.from_position(q, joint_names=franka_kinematics.joint_names)
        )
        robot_spheres = kin_state.robot_spheres

        result = cost.forward(robot_spheres)
        assert result.shape == (batch_size, horizon, 1)
        # The cost should be relatively small for default config
        assert torch.all(result <= 0.1)

    def test_validate_input_wrong_ndim(self, device_cfg, self_collision_kin_config):
        """Test validation with wrong number of dimensions."""
        cfg = SelfCollisionCostCfg(
            weight=1.0,
            device_cfg=device_cfg,
            self_collision_kin_config=self_collision_kin_config,
        )
        cost = SelfCollisionCost(cfg)
        batch_size = 4
        horizon = 10
        cost.setup_batch_tensors(batch_size, horizon)

        # Wrong ndim (3 instead of 4)
        robot_spheres = torch.zeros(
            (batch_size, horizon, 4), **device_cfg.as_torch_dict()
        )

        with pytest.raises(Exception):
            cost.forward(robot_spheres)

    def test_validate_input_wrong_batch_size(
        self, device_cfg, self_collision_kin_config
    ):
        """Test validation with wrong batch size."""
        cfg = SelfCollisionCostCfg(
            weight=1.0,
            device_cfg=device_cfg,
            self_collision_kin_config=self_collision_kin_config,
        )
        cost = SelfCollisionCost(cfg)
        batch_size = 4
        horizon = 10
        num_spheres = self_collision_kin_config.num_spheres
        cost.setup_batch_tensors(batch_size, horizon)

        # Wrong batch size (8 instead of 4)
        robot_spheres = torch.zeros(
            (8, horizon, num_spheres, 4), **device_cfg.as_torch_dict()
        )

        with pytest.raises(Exception):
            cost.forward(robot_spheres)

    def test_validate_input_wrong_horizon(self, device_cfg, self_collision_kin_config):
        """Test validation with wrong horizon."""
        cfg = SelfCollisionCostCfg(
            weight=1.0,
            device_cfg=device_cfg,
            self_collision_kin_config=self_collision_kin_config,
        )
        cost = SelfCollisionCost(cfg)
        batch_size = 4
        horizon = 10
        num_spheres = self_collision_kin_config.num_spheres
        cost.setup_batch_tensors(batch_size, horizon)

        # Wrong horizon (20 instead of 10)
        robot_spheres = torch.zeros(
            (batch_size, 20, num_spheres, 4), **device_cfg.as_torch_dict()
        )

        with pytest.raises(Exception):
            cost.forward(robot_spheres)

    def test_validate_input_wrong_num_spheres(
        self, device_cfg, self_collision_kin_config
    ):
        """Test validation with wrong number of spheres."""
        cfg = SelfCollisionCostCfg(
            weight=1.0,
            device_cfg=device_cfg,
            self_collision_kin_config=self_collision_kin_config,
        )
        cost = SelfCollisionCost(cfg)
        batch_size = 4
        horizon = 10
        cost.setup_batch_tensors(batch_size, horizon)

        # Wrong num_spheres (10 instead of actual)
        robot_spheres = torch.zeros(
            (batch_size, horizon, 10, 4), **device_cfg.as_torch_dict()
        )

        with pytest.raises(Exception):
            cost.forward(robot_spheres)

    def test_reset(self, device_cfg, self_collision_kin_config):
        """Test reset method."""
        cfg = SelfCollisionCostCfg(
            weight=1.0,
            device_cfg=device_cfg,
            self_collision_kin_config=self_collision_kin_config,
        )
        cost = SelfCollisionCost(cfg)
        cost.setup_batch_tensors(batch_size=4, horizon=10)

        # Reset should not raise
        cost.reset()

    def test_convert_to_binary(self, device_cfg, self_collision_kin_config, franka_kinematics):
        """Test convert_to_binary option."""
        cfg = SelfCollisionCostCfg(
            weight=1.0,
            device_cfg=device_cfg,
            self_collision_kin_config=self_collision_kin_config,
            convert_to_binary=True,
        )
        cost = SelfCollisionCost(cfg)
        batch_size = 4
        horizon = 10
        dof = franka_kinematics.dof
        cost.setup_batch_tensors(batch_size, horizon)

        # Create random joint configurations
        q = torch.zeros((batch_size, horizon, dof), **device_cfg.as_torch_dict())

        kin_state = franka_kinematics.compute_kinematics(
            JointState.from_position(q, joint_names=franka_kinematics.joint_names)
        )
        robot_spheres = kin_state.robot_spheres

        result = cost.forward(robot_spheres)
        assert result.shape == (batch_size, horizon, 1)


class TestSelfCollisionKinematicsCfg:
    """Test SelfCollisionKinematicsCfg properties and methods."""

    def test_num_spheres(self, self_collision_kin_config):
        """Test num_spheres property."""
        assert self_collision_kin_config.num_spheres > 0

    def test_collision_pairs(self, self_collision_kin_config):
        """Test collision_pairs is set correctly."""
        assert self_collision_kin_config.collision_pairs is not None
        assert self_collision_kin_config.collision_pairs.shape[1] == 2

    def test_sphere_padding(self, self_collision_kin_config):
        """Test sphere_padding is set."""
        assert self_collision_kin_config.sphere_padding is not None
        assert (
            self_collision_kin_config.sphere_padding.shape[0]
            == self_collision_kin_config.num_spheres
        )

    def test_num_blocks_per_batch(self, self_collision_kin_config):
        """Test num_blocks_per_batch property."""
        num_blocks = self_collision_kin_config.num_blocks_per_batch
        assert num_blocks >= 1

    def test_max_threads_per_block(self, self_collision_kin_config):
        """Test max_threads_per_block property."""
        max_threads = self_collision_kin_config.max_threads_per_block
        assert max_threads > 0


class TestSelfCollisionCostGradients:
    """Test gradient computation for SelfCollisionCost.

    Note: SelfCollisionCost uses CUDA kernels that return pre-computed gradients.
    The gradient is stored in the cost's internal buffer (_out_grad) and is returned
    during backward pass when use_grad_input=True.
    """

    def test_self_collision_cost_gradient_buffer(
        self, device_cfg, self_collision_kin_config, franka_kinematics
    ):
        """Test that gradient buffer is populated after forward pass."""
        cfg = SelfCollisionCostCfg(
            weight=1.0,
            device_cfg=device_cfg,
            self_collision_kin_config=self_collision_kin_config,
            use_grad_input=True,  # Required for gradient computation
        )
        cost = SelfCollisionCost(cfg)
        batch_size = 4
        horizon = 10
        dof = franka_kinematics.dof
        cost.setup_batch_tensors(batch_size, horizon)

        # Create joint configurations
        q = torch.zeros((batch_size, horizon, dof), **device_cfg.as_torch_dict())
        kin_state = franka_kinematics.compute_kinematics(
            JointState.from_position(q, joint_names=franka_kinematics.joint_names)
        )

        robot_spheres = kin_state.robot_spheres.clone()

        # Forward pass
        result = cost.forward(robot_spheres)

        # Verify gradient buffer exists and has correct shape
        assert cost._out_grad is not None, "Gradient buffer should exist"
        assert cost._out_grad.shape[0] == batch_size, "Gradient buffer batch dimension mismatch"
        assert cost._out_grad.shape[1] == horizon, "Gradient buffer horizon dimension mismatch"

    def test_self_collision_cost_forward_returns_tensor(
        self, device_cfg, self_collision_kin_config, franka_kinematics
    ):
        """Test that forward returns a proper tensor that can be used in loss computation."""
        cfg = SelfCollisionCostCfg(
            weight=1.0,
            device_cfg=device_cfg,
            self_collision_kin_config=self_collision_kin_config,
            use_grad_input=True,
        )
        cost = SelfCollisionCost(cfg)
        batch_size = 4
        horizon = 10
        dof = franka_kinematics.dof
        cost.setup_batch_tensors(batch_size, horizon)

        # Create joint configurations
        q = torch.zeros((batch_size, horizon, dof), **device_cfg.as_torch_dict())
        kin_state = franka_kinematics.compute_kinematics(
            JointState.from_position(q, joint_names=franka_kinematics.joint_names)
        )

        robot_spheres = kin_state.robot_spheres

        # Forward pass
        result = cost.forward(robot_spheres)

        # Verify result is a tensor with correct shape
        assert isinstance(result, torch.Tensor), "Result should be a tensor"
        assert result.shape == (batch_size, horizon, 1), "Result shape mismatch"
        assert torch.isfinite(result).all(), "Result contains non-finite values"

