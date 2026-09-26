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
from curobo._src.robot.types.self_collision_params import SelfCollisionKinematicsCfg
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


class TestSelfCollisionCostBackward:
    """Check per-query center derivatives through the production CUDA cost."""

    @staticmethod
    def make_pair(
        device_cfg: DeviceCfg,
        batch_size: int,
        horizon: int,
        num_spheres: int = 2,
        use_grad_input: bool = True,
        collision_pairs: tuple[tuple[int, int], ...] = ((0, 1),),
    ) -> tuple[SelfCollisionCost, torch.Tensor]:
        """Create one checked pair with fixed radii and optional ignored spheres."""
        config = SelfCollisionCostCfg(
            weight=1.0,
            device_cfg=device_cfg,
            use_grad_input=use_grad_input,
            self_collision_kin_config=SelfCollisionKinematicsCfg(
                num_spheres=num_spheres,
                sphere_padding=torch.zeros(num_spheres, **device_cfg.as_torch_dict()),
                collision_pairs=torch.tensor(
                    collision_pairs, device=device_cfg.device, dtype=torch.int16
                ),
            ),
        )
        cost = SelfCollisionCost(config)
        cost.setup_batch_tensors(batch_size, horizon)
        spheres = torch.zeros(
            (batch_size, horizon, num_spheres, 4), **device_cfg.as_torch_dict()
        )
        spheres[..., 3] = 0.5
        spheres[..., 1, :3] = device_cfg.to_device([0.5, 0.125, -0.25])
        return cost, spheres.requires_grad_(True)

    @staticmethod
    def expected_pair(
        spheres: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute fixed-radius pair costs and XYZ derivatives with scalar arithmetic."""
        batch_size, horizon, num_spheres, _ = spheres.shape
        expected_cost = torch.zeros((batch_size, horizon, 1), dtype=torch.float32)
        expected_grad = torch.zeros((batch_size, horizon, num_spheres, 3), dtype=torch.float32)
        centers = spheres.detach().cpu().tolist()
        scales = weights.cpu().tolist()
        for batch in range(batch_size):
            for time in range(horizon):
                delta = [
                    centers[batch][time][1][axis] - centers[batch][time][0][axis]
                    for axis in range(3)
                ]
                value = 0.5 * max(1.0 - sum(component**2 for component in delta), 0.0)
                expected_cost[batch, time, 0] = value
                if value > 0.0:
                    for axis in range(3):
                        derivative = delta[axis] * scales[batch][time][0]
                        expected_grad[batch, time, 0, axis] = derivative
                        expected_grad[batch, time, 1, axis] = -derivative
        return expected_cost, expected_grad

    @pytest.mark.parametrize(
        "batch_size,horizon,num_spheres",
        [(1, 1, 2), (2, 3, 2), (1, 3, 5), (2, 2, 2), (1, 2, 2), (2, 1, 2)],
    )
    def test_weighted_center_gradients(
        self, device_cfg: DeviceCfg, batch_size: int, horizon: int, num_spheres: int
    ) -> None:
        """Keep incoming derivatives aligned with their batch and time indices."""
        cost, spheres = self.make_pair(device_cfg, batch_size, horizon, num_spheres)
        weights = torch.arange(
            1, batch_size * horizon + 1, **device_cfg.as_torch_dict()
        ).reshape(batch_size, horizon, 1)
        expected_cost, expected_grad = self.expected_pair(spheres, weights)
        distance = cost.forward(spheres)
        gradient = torch.autograd.grad(distance, spheres, grad_outputs=weights)[0]
        torch.testing.assert_close(distance.detach().cpu(), expected_cost)
        torch.testing.assert_close(gradient[..., :3].cpu(), expected_grad)

    @pytest.mark.parametrize("strided", [False, True])
    def test_signed_zero_and_strided_weights(self, device_cfg: DeviceCfg, strided: bool) -> None:
        """Honor zero and negative weights, including a noncontiguous upstream tensor."""
        cost, spheres = self.make_pair(device_cfg, 2, 2)
        weights = device_cfg.to_device([[0.0, 9.0, -2.0, 9.0], [3.0, 9.0, -4.0, 9.0]])
        weights = weights[:, ::2].unsqueeze(-1)
        if not strided:
            weights = weights.contiguous()
        assert weights.is_contiguous() != strided
        _, expected = self.expected_pair(spheres, weights)
        gradient = torch.autograd.grad(cost.forward(spheres), spheres, grad_outputs=weights)[0]
        torch.testing.assert_close(gradient[..., :3].cpu(), expected)

    def test_query_specific_active_pairs(self, device_cfg: DeviceCfg) -> None:
        """Weight the selected pair independently when it changes between queries."""
        cost, spheres = self.make_pair(
            device_cfg, 2, 2, num_spheres=4, collision_pairs=((0, 1), (2, 3))
        )
        weights = device_cfg.to_device([1.0, 2.0, 3.0, 4.0]).reshape(2, 2, 1)
        expected = torch.zeros((2, 2, 4, 3), dtype=torch.float32)
        delta = [0.5, 0.125, -0.25]
        with torch.no_grad():
            for batch in range(2):
                for time in range(2):
                    active = 2 * ((batch + time) % 2)
                    inactive = 2 - active
                    spheres[batch, time, active, :3] = 0.0
                    spheres[batch, time, active + 1, :3] = device_cfg.to_device(delta)
                    spheres[batch, time, inactive, :3] = device_cfg.to_device([10, 0, 0])
                    spheres[batch, time, inactive + 1, :3] = device_cfg.to_device([12, 0, 0])
                    for axis in range(3):
                        derivative = delta[axis] * (batch * 2 + time + 1)
                        expected[batch, time, active, axis] = derivative
                        expected[batch, time, active + 1, axis] = -derivative
        distance = cost.forward(spheres)
        gradient = torch.autograd.grad(distance, spheres, grad_outputs=weights)[0]
        expected_cost = torch.full((2, 2, 1), 0.5 * (1 - sum(x**2 for x in delta)))
        torch.testing.assert_close(distance.detach().cpu(), expected_cost)
        torch.testing.assert_close(gradient[..., :3].cpu(), expected)

    @pytest.mark.parametrize("batch_size,horizon", [(2, 3), (2, 2)])
    def test_sum_fast_path(self, device_cfg: DeviceCfg, batch_size: int, horizon: int) -> None:
        """Preserve the documented sum-only path when incoming scaling is disabled."""
        cost, spheres = self.make_pair(device_cfg, batch_size, horizon, use_grad_input=False)
        weights = torch.ones((batch_size, horizon, 1), **device_cfg.as_torch_dict())
        _, expected = self.expected_pair(spheres, weights)
        gradient = torch.autograd.grad(cost.forward(spheres).sum(), spheres)[0]
        torch.testing.assert_close(gradient[..., :3].cpu(), expected)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_center_finite_difference(self, device_cfg: DeviceCfg, axis: int) -> None:
        """Check XYZ derivatives away from pair-selection and contact boundaries."""
        cost, spheres = self.make_pair(device_cfg, 2, 2)
        weights = device_cfg.to_device([1.0, 2.0, 3.0, 4.0]).reshape(2, 2, 1)
        gradient = torch.autograd.grad(cost.forward(spheres), spheres, grad_outputs=weights)[0]
        derivative = gradient[1, 0, 0, axis].item()
        epsilon = 1e-3
        plus, minus = spheres.detach().clone(), spheres.detach().clone()
        plus[1, 0, 0, axis] += epsilon
        minus[1, 0, 0, axis] -= epsilon
        upper = (cost.forward(plus) * weights).sum().item()
        lower = (cost.forward(minus) * weights).sum().item()
        assert derivative == pytest.approx((upper - lower) / (2 * epsilon), abs=2e-4)

    def test_repeated_collision_and_separation(self, device_cfg: DeviceCfg) -> None:
        """Clear separated gradients and preserve previously returned weighted gradients."""
        cost, spheres = self.make_pair(device_cfg, 2, 2)
        weights = device_cfg.to_device([1.0, 2.0, 3.0, 4.0]).reshape(2, 2, 1)
        previous = []
        for separation in (0.5, 1.5, 0.25):
            with torch.no_grad():
                spheres[..., 1, 0] = separation
            expected_cost, expected_grad = self.expected_pair(spheres, weights)
            distance = cost.forward(spheres)
            gradient = torch.autograd.grad(distance, spheres, grad_outputs=weights)[0]
            torch.testing.assert_close(distance.detach().cpu(), expected_cost)
            torch.testing.assert_close(gradient[..., :3].cpu(), expected_grad)
            for retained, snapshot in previous:
                torch.testing.assert_close(retained, snapshot)
            previous.append((gradient, gradient.clone()))

    def test_cuda_graph_replay(self, device_cfg: DeviceCfg) -> None:
        """Replay captured forward/backward with changed geometry and incoming weights."""
        cost, spheres = self.make_pair(device_cfg, 2, 3)
        weights = torch.ones((2, 3, 1), **device_cfg.as_torch_dict())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                torch.autograd.grad(cost.forward(spheres), spheres, grad_outputs=weights)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            distance = cost.forward(spheres)
            gradient = torch.autograd.grad(distance, spheres, grad_outputs=weights)[0]
        for separation, scale in ((0.5, 1.0), (1.5, -2.0), (0.25, 3.0)):
            with torch.no_grad():
                spheres[..., 1, 0] = separation
                weights.copy_(
                    torch.arange(6, **device_cfg.as_torch_dict()).reshape(2, 3, 1) * scale
                )
            expected_cost, expected_grad = self.expected_pair(spheres, weights)
            graph.replay()
            torch.testing.assert_close(distance.detach().cpu(), expected_cost)
            torch.testing.assert_close(gradient[..., :3].cpu(), expected_grad)
