# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for SelfCollisionCost and SelfCollisionCostCfg using real Franka kinematics."""

# Standard Library
import copy

# Third Party
import pytest
import torch

# CuRobo
import curobo._src.runtime as curobo_runtime
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


@pytest.fixture(scope="module")
def large_aabb_collision_config(device_cfg: DeviceCfg) -> SelfCollisionKinematicsCfg:
    """Create a large topology that selects the link-AABB path."""
    num_spheres = 4
    num_pairs = 40_705
    collision_pairs = torch.tensor([[0, 1]], device=device_cfg.device, dtype=torch.int16).repeat(
        num_pairs, 1
    )
    collision_cfg = SelfCollisionKinematicsCfg(
        num_spheres=num_spheres,
        sphere_padding=torch.zeros(num_spheres, **device_cfg.as_torch_dict()),
        collision_pairs=collision_pairs,
    )
    collision_cfg.configure_link_aabb(
        torch.arange(num_spheres, device=device_cfg.device, dtype=torch.int16)
    )
    assert collision_cfg.use_link_aabb
    return collision_cfg


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

    def test_sum_collisions_rejects_pair_distance_storage(
        self, device_cfg, self_collision_kin_config
    ):
        """Summed costs do not retain one distance per collision pair."""
        with pytest.raises(ValueError, match="pair-distance storage"):
            SelfCollisionCost(
                SelfCollisionCostCfg(
                    weight=1.0,
                    device_cfg=device_cfg,
                    self_collision_kin_config=self_collision_kin_config,
                    store_pair_distance=True,
                    sum_collisions=True,
                )
            )


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

    def test_aabb_selected_at_shared_memory_limit(
        self,
        device_cfg: DeviceCfg,
        large_aabb_collision_config: SelfCollisionKinematicsCfg,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AABB remains selected when its total shared memory exactly fits."""
        from curobo._src.curobolib.backends.cuda_core_backend import geometry

        aabb_shared_memory = geometry.get_self_collision_shared_memory_size(
            large_aabb_collision_config.num_spheres,
            large_aabb_collision_config.num_collision_links,
        )
        monkeypatch.setattr(
            geometry,
            "get_device_max_shared_memory",
            lambda _device: aabb_shared_memory,
        )

        cost = SelfCollisionCost(
            SelfCollisionCostCfg(
                weight=1.0,
                device_cfg=device_cfg,
                self_collision_kin_config=large_aabb_collision_config,
            )
        )
        cost.setup_batch_tensors(batch_size=1, horizon=1)

        assert cost._use_link_aabb
        robot_spheres = torch.zeros((1, 1, 4, 4), **device_cfg.as_torch_dict())
        robot_spheres[0, 0, 0, 3] = 0.2
        robot_spheres[0, 0, 1] = torch.tensor([0.1, 0.0, 0.0, 0.2], **device_cfg.as_torch_dict())
        assert cost.forward(robot_spheres).item() > 0.0

    def test_aabb_falls_back_to_exact_before_buffer_allocation(
        self,
        device_cfg: DeviceCfg,
        large_aabb_collision_config: SelfCollisionKinematicsCfg,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An unsupported AABB allocation should warn, select, and size the exact path."""
        from curobo._src.curobolib.backends.cuda_core_backend import geometry

        exact_shared_memory = geometry.get_self_collision_shared_memory_size(
            large_aabb_collision_config.num_spheres
        )
        monkeypatch.setattr(
            geometry,
            "get_device_max_shared_memory",
            lambda _device: exact_shared_memory,
        )

        with caplog.at_level("WARNING", logger="curobo"):
            cost = SelfCollisionCost(
                SelfCollisionCostCfg(
                    weight=1.0,
                    device_cfg=device_cfg,
                    self_collision_kin_config=large_aabb_collision_config,
                )
            )
        cost.setup_batch_tensors(batch_size=1, horizon=1)

        assert not cost._use_link_aabb
        assert "too many spheres or collision links" in caplog.text
        assert "Falling back to exact self collision" in caplog.text
        robot_spheres = torch.zeros((1, 1, 4, 4), **device_cfg.as_torch_dict())
        robot_spheres[0, 0, 0, 3] = 0.2
        robot_spheres[0, 0, 1] = torch.tensor([0.1, 0.0, 0.0, 0.2], **device_cfg.as_torch_dict())
        assert cost.forward(robot_spheres).item() > 0.0

    def test_aabb_preflight_raises_when_exact_does_not_fit(
        self,
        device_cfg: DeviceCfg,
        large_aabb_collision_config: SelfCollisionKinematicsCfg,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A topology unsupported by both kernels should fail before allocation."""
        from curobo._src.curobolib.backends.cuda_core_backend import geometry

        exact_shared_memory = geometry.get_self_collision_shared_memory_size(
            large_aabb_collision_config.num_spheres
        )
        monkeypatch.setattr(
            geometry,
            "get_device_max_shared_memory",
            lambda _device: exact_shared_memory - 1,
        )

        with pytest.raises(ValueError, match="exact kernel"):
            SelfCollisionCost(
                SelfCollisionCostCfg(
                    weight=1.0,
                    device_cfg=device_cfg,
                    self_collision_kin_config=large_aabb_collision_config,
                )
            )

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
        default_position = (
            franka_robot_cfg.kinematics.kinematics_config.cspace.default_joint_position
        )
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
        robot_spheres = torch.zeros((batch_size, horizon, 4), **device_cfg.as_torch_dict())

        with pytest.raises(Exception):
            cost.forward(robot_spheres)

    def test_validate_input_wrong_batch_size(self, device_cfg, self_collision_kin_config):
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
        robot_spheres = torch.zeros((8, horizon, num_spheres, 4), **device_cfg.as_torch_dict())

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
        robot_spheres = torch.zeros((batch_size, 20, num_spheres, 4), **device_cfg.as_torch_dict())

        with pytest.raises(Exception):
            cost.forward(robot_spheres)

    def test_validate_input_wrong_num_spheres(self, device_cfg, self_collision_kin_config):
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
        robot_spheres = torch.zeros((batch_size, horizon, 10, 4), **device_cfg.as_torch_dict())

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

    def test_max_threads_per_block(self, self_collision_kin_config):
        """Test max_threads_per_block property."""
        max_threads = self_collision_kin_config.max_threads_per_block
        assert max_threads > 0

    def test_default_launch_geometry(self):
        """Select a warp-rounded block width from the exact pair count."""
        expected_widths = {
            0: 32,
            1: 32,
            192: 64,
            193: 96,
            694: 256,
            1_536: 256,
            1_537: 256,
            162_111: 256,
        }
        for num_pairs, expected_width in expected_widths.items():
            collision_cfg = SelfCollisionKinematicsCfg(
                collision_pairs=torch.zeros((num_pairs, 2), dtype=torch.int16)
            )
            assert collision_cfg.max_threads_per_block == expected_width

    def test_small_gpu_launch_policy_override(self, monkeypatch):
        """Allow the embedded launch policy to be selected on a larger GPU."""
        monkeypatch.setattr(curobo_runtime, "kernel_launch_sm_count", 32)
        collision_cfg = SelfCollisionKinematicsCfg(
            collision_pairs=torch.zeros((694, 2), device="cuda", dtype=torch.int16)
        )
        assert collision_cfg.max_threads_per_block == 64


class TestSelfCollisionCostGradients:
    """Test gradient computation for SelfCollisionCost.

    Note: SelfCollisionCost uses CUDA kernels that return pre-computed gradients.
    The gradient is stored in the cost's internal buffer (_out_grad) and is returned
    during backward pass when use_grad_input=True.
    """

    @pytest.mark.parametrize("use_link_aabb", [False, True])
    def test_batched_upstream_gradient_scaling(
        self, device_cfg: DeviceCfg, use_link_aabb: bool
    ) -> None:
        """Scale every batch/horizon sphere gradient by its own upstream value."""
        num_spheres = 3
        collision_pairs = torch.tensor(
            [[0, 1], [0, 2]], device=device_cfg.device, dtype=torch.int16
        ).repeat((501, 1))
        collision_cfg = SelfCollisionKinematicsCfg(
            num_spheres=num_spheres,
            sphere_padding=torch.zeros(num_spheres, **device_cfg.as_torch_dict()),
            collision_pairs=collision_pairs,
            _use_link_aabb=use_link_aabb,
        )
        collision_cfg.configure_link_aabb(
            torch.arange(num_spheres, device=device_cfg.device, dtype=torch.int16)
        )
        assert collision_cfg.use_link_aabb == use_link_aabb

        batch_size = 2
        horizon = 3
        cost = SelfCollisionCost(
            SelfCollisionCostCfg(
                weight=1.0,
                device_cfg=device_cfg,
                self_collision_kin_config=collision_cfg,
                use_grad_input=True,
            )
        )
        cost.setup_batch_tensors(batch_size=batch_size, horizon=horizon)
        spheres = torch.tensor(
            [[0.0, 0.0, 0.0, 0.2],
             [0.1, 0.0, 0.0, 0.2],
             [0.2, 0.0, 0.0, 0.2]],
            **device_cfg.as_torch_dict(),
        ).view(1, 1, num_spheres, 4).expand(batch_size, horizon, -1, -1).clone()
        spheres.requires_grad_(True)
        distance = cost.forward(spheres)
        kernel_gradient = cost._out_grad.detach().clone()
        upstream_gradient = torch.tensor(
            [[[0.25], [0.5], [1.0]], [[1.5], [2.0], [3.0]]],
            **device_cfg.as_torch_dict(),
        )

        gradient = torch.autograd.grad(
            distance, spheres, grad_outputs=upstream_gradient
        )[0]

        torch.testing.assert_close(
            gradient,
            kernel_gradient * upstream_gradient.unsqueeze(-1),
        )

    def test_sum_collision_exact_and_aabb_match(self, device_cfg: DeviceCfg) -> None:
        """Sum every positive pair cost and gradient in exact and link-AABB kernels."""
        num_spheres = 3
        pair_repeats = 501
        collision_pairs = torch.tensor(
            [[0, 1], [0, 2]], device=device_cfg.device, dtype=torch.int16
        ).repeat((pair_repeats, 1))
        collision_cfg = SelfCollisionKinematicsCfg(
            num_spheres=num_spheres,
            sphere_padding=torch.zeros(num_spheres, **device_cfg.as_torch_dict()),
            collision_pairs=collision_pairs,
        )
        collision_cfg.configure_link_aabb(
            torch.arange(num_spheres, device=device_cfg.device, dtype=torch.int16)
        )
        assert collision_cfg.use_link_aabb

        aabb_cost = SelfCollisionCost(
            SelfCollisionCostCfg(
                weight=1.0,
                device_cfg=device_cfg,
                self_collision_kin_config=collision_cfg,
                use_grad_input=True,
                sum_collisions=True,
            )
        )
        aabb_cost.setup_batch_tensors(batch_size=1, horizon=1)
        spheres = torch.tensor(
            [[[[0.0, 0.0, 0.0, 0.2], [0.1, 0.0, 0.0, 0.2], [0.2, 0.0, 0.0, 0.2]]]],
            **device_cfg.as_torch_dict(),
        ).requires_grad_(True)
        aabb_distance = aabb_cost.forward(spheres)
        aabb_gradient = torch.autograd.grad(aabb_distance.sum(), spheres)[0]

        exact_collision_cfg = copy.copy(collision_cfg)
        exact_collision_cfg._use_link_aabb = False
        exact_cost = SelfCollisionCost(
            SelfCollisionCostCfg(
                weight=1.0,
                device_cfg=device_cfg,
                self_collision_kin_config=exact_collision_cfg,
                use_grad_input=True,
                sum_collisions=True,
            )
        )
        exact_cost.setup_batch_tensors(batch_size=1, horizon=1)
        exact_spheres = spheres.detach().clone().requires_grad_(True)
        exact_distance = exact_cost.forward(exact_spheres)
        exact_gradient = torch.autograd.grad(exact_distance.sum(), exact_spheres)[0]

        expected_distance = pair_repeats * 0.5 * ((0.4**2 - 0.1**2) + (0.4**2 - 0.2**2))
        expected_gradient = torch.tensor(
            [
                [
                    [
                        [0.3 * pair_repeats, 0.0, 0.0, -1.0],
                        [-0.1 * pair_repeats, 0.0, 0.0, -1.0],
                        [-0.2 * pair_repeats, 0.0, 0.0, -1.0],
                    ]
                ]
            ],
            **device_cfg.as_torch_dict(),
        )
        assert torch.allclose(aabb_distance, exact_distance, rtol=1e-5, atol=1e-3)
        assert torch.allclose(aabb_gradient, exact_gradient, rtol=1e-5, atol=1e-3)
        assert torch.allclose(
            aabb_distance, torch.tensor([[[expected_distance]]], **device_cfg.as_torch_dict())
        )
        assert torch.allclose(aabb_gradient, expected_gradient, rtol=1e-5, atol=1e-3)

    def test_per_sphere_max_collision_exact_and_aabb_match(self, device_cfg: DeviceCfg) -> None:
        """Select one positive contact per sphere in exact and link-AABB kernels."""
        num_spheres = 3
        pair_repeats = 501
        collision_pairs = torch.tensor(
            [[0, 1], [0, 2]], device=device_cfg.device, dtype=torch.int16
        ).repeat((pair_repeats, 1))
        collision_cfg = SelfCollisionKinematicsCfg(
            num_spheres=num_spheres,
            sphere_padding=torch.zeros(num_spheres, **device_cfg.as_torch_dict()),
            collision_pairs=collision_pairs,
        )
        collision_cfg.configure_link_aabb(
            torch.arange(num_spheres, device=device_cfg.device, dtype=torch.int16)
        )
        aabb_cost = SelfCollisionCost(
            SelfCollisionCostCfg(
                weight=1.0,
                device_cfg=device_cfg,
                self_collision_kin_config=collision_cfg,
                use_grad_input=True,
                per_sphere_max_collisions=True,
            )
        )
        aabb_cost.setup_batch_tensors(batch_size=1, horizon=1)
        spheres = torch.tensor(
            [[[[0.0, 0.0, 0.0, 0.2], [0.1, 0.0, 0.0, 0.2], [0.2, 0.0, 0.0, 0.2]]]],
            **device_cfg.as_torch_dict(),
        ).requires_grad_(True)
        aabb_distance = aabb_cost.forward(spheres)
        aabb_gradient = torch.autograd.grad(aabb_distance.sum(), spheres)[0]

        exact_collision_cfg = copy.copy(collision_cfg)
        exact_collision_cfg._use_link_aabb = False
        exact_cost = SelfCollisionCost(
            SelfCollisionCostCfg(
                weight=1.0,
                device_cfg=device_cfg,
                self_collision_kin_config=exact_collision_cfg,
                use_grad_input=True,
                per_sphere_max_collisions=True,
            )
        )
        exact_cost.setup_batch_tensors(batch_size=1, horizon=1)
        exact_spheres = spheres.detach().clone().requires_grad_(True)
        exact_distance = exact_cost.forward(exact_spheres)
        exact_gradient = torch.autograd.grad(exact_distance.sum(), exact_spheres)[0]

        expected_distance = 0.5 * (2.0 * (0.4**2 - 0.1**2) + (0.4**2 - 0.2**2))
        expected_gradient = torch.tensor(
            [[[[0.4, 0.0, 0.0, -1.0],
              [-0.2, 0.0, 0.0, -1.0],
              [-0.2, 0.0, 0.0, -1.0]]]],
            **device_cfg.as_torch_dict(),
        )
        assert torch.allclose(aabb_distance, exact_distance, rtol=1e-5, atol=1e-3)
        assert torch.allclose(aabb_gradient, exact_gradient, rtol=1e-5, atol=1e-3)
        assert torch.allclose(
            aabb_distance, torch.tensor([[[expected_distance]]], **device_cfg.as_torch_dict())
        )
        assert torch.allclose(aabb_gradient, expected_gradient, rtol=1e-5, atol=1e-3)

    def test_link_aabb_forward_backward_and_sparse_clear(self, device_cfg):
        """Validate link-AABB results and clearing of prior sparse gradients."""
        num_spheres = 674
        first_sphere = 300
        second_sphere = 301
        num_pairs = 40_705
        collision_pairs = torch.tensor(
            [[first_sphere, second_sphere]], device=device_cfg.device, dtype=torch.int16
        ).repeat(num_pairs, 1)
        collision_cfg = SelfCollisionKinematicsCfg(
            num_spheres=num_spheres,
            sphere_padding=torch.zeros(num_spheres, **device_cfg.as_torch_dict()),
            collision_pairs=collision_pairs,
        )
        sphere_link_map = torch.zeros(num_spheres, device=device_cfg.device, dtype=torch.int16)
        sphere_link_map[second_sphere] = 1
        collision_cfg.configure_link_aabb(sphere_link_map)
        assert collision_cfg.use_link_aabb

        cfg = SelfCollisionCostCfg(
            weight=1.0,
            device_cfg=device_cfg,
            self_collision_kin_config=collision_cfg,
            use_grad_input=True,
        )
        cost = SelfCollisionCost(cfg)
        cost.setup_batch_tensors(batch_size=1, horizon=1)

        colliding_spheres = torch.zeros((1, 1, num_spheres, 4), **device_cfg.as_torch_dict())
        colliding_spheres[0, 0, first_sphere] = torch.tensor(
            [0.0, 0.0, 0.0, 0.2], **device_cfg.as_torch_dict()
        )
        colliding_spheres[0, 0, second_sphere] = torch.tensor(
            [0.1, 0.0, 0.0, 0.2], **device_cfg.as_torch_dict()
        )
        colliding_spheres.requires_grad_(True)

        result = cost.forward(colliding_spheres)
        gradient = torch.autograd.grad(result.sum(), colliding_spheres)[0]

        exact_collision_cfg = copy.copy(collision_cfg)
        exact_collision_cfg._use_link_aabb = False
        exact_cost = SelfCollisionCost(
            SelfCollisionCostCfg(
                weight=1.0,
                device_cfg=device_cfg,
                self_collision_kin_config=exact_collision_cfg,
                use_grad_input=True,
            )
        )
        exact_cost.setup_batch_tensors(batch_size=1, horizon=1)
        exact_spheres = colliding_spheres.detach().clone().requires_grad_(True)
        exact_result = exact_cost.forward(exact_spheres)
        exact_gradient = torch.autograd.grad(exact_result.sum(), exact_spheres)[0]

        assert torch.allclose(result, exact_result)
        assert torch.allclose(gradient, exact_gradient)
        assert torch.allclose(result, torch.tensor([[[0.075]]], **device_cfg.as_torch_dict()))
        assert torch.allclose(
            gradient[0, 0, first_sphere],
            torch.tensor([0.1, 0.0, 0.0, -1.0], **device_cfg.as_torch_dict()),
        )
        assert torch.allclose(
            gradient[0, 0, second_sphere],
            torch.tensor([-0.1, 0.0, 0.0, -1.0], **device_cfg.as_torch_dict()),
        )

        separated_spheres = colliding_spheres.detach().clone()
        separated_spheres[0, 0, second_sphere, 0] = 1.0
        separated_spheres.requires_grad_(True)

        separated_result = cost.forward(separated_spheres)
        separated_gradient = torch.autograd.grad(separated_result.sum(), separated_spheres)[0]
        assert torch.count_nonzero(separated_result) == 0
        assert torch.count_nonzero(separated_gradient) == 0

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
        cost.forward(robot_spheres)

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
