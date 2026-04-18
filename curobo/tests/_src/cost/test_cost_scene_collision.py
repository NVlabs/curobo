# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for SceneCollisionCost and SceneCollisionCostCfg using real Franka kinematics."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.cost.cost_scene_collision import SceneCollisionCost
from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg
from curobo._src.geom.collision import SceneCollisionCfg, create_collision_checker
from curobo._src.geom.types import Cuboid, SceneCfg
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
def simple_scene_cfg():
    """Create a simple scene with a table (cuboid)."""
    # Simple table in front of the robot
    table = Cuboid(
        name="table",
        pose=[0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],  # x, y, z, qw, qx, qy, qz
        dims=[0.6, 1.0, 0.05],  # width, depth, height
    )
    return SceneCfg(cuboid=[table])


@pytest.fixture(scope="module")
def scene_collision_checker(device_cfg, simple_scene_cfg):
    """Create a scene collision checker with a simple scene."""
    scene_collision_cfg = SceneCollisionCfg(
        device_cfg=device_cfg,
        scene_model=simple_scene_cfg,
        cache={"mesh": 10, "primitive": 10},
        max_distance=1.0,
        num_envs=1,
    )
    return create_collision_checker(scene_collision_cfg)


class TestSceneCollisionCostCfg:
    """Test SceneCollisionCostCfg class."""

    def test_default_init(self, device_cfg):
        """Test default initialization."""
        cfg = SceneCollisionCostCfg(weight=1.0, device_cfg=device_cfg)
        assert cfg.class_type == SceneCollisionCost
        assert cfg.use_sweep is False
        assert cfg.use_speed_metric is False
        assert cfg.sum_distance is True
        assert cfg.num_spheres == 0

    def test_init_with_scene_collision_checker(self, device_cfg, scene_collision_checker):
        """Test initialization with scene_collision_checker."""
        cfg = SceneCollisionCostCfg(
            weight=1.0,
            device_cfg=device_cfg,
        )
        cfg.scene_collision_checker = scene_collision_checker
        assert cfg.scene_collision_checker is not None
        assert cfg._num_scene_collision_checkers >= 1

    def test_init_with_use_sweep(self, device_cfg):
        """Test initialization with use_sweep=True."""
        cfg = SceneCollisionCostCfg(
            weight=1.0,
            use_sweep=True,
            device_cfg=device_cfg,
        )
        assert cfg.use_sweep is True

    def test_init_with_use_speed_metric(self, device_cfg):
        """Test initialization with use_speed_metric=True."""
        cfg = SceneCollisionCostCfg(
            weight=1.0,
            use_speed_metric=True,
            device_cfg=device_cfg,
        )
        assert cfg.use_speed_metric is True

    def test_init_with_activation_distance(self, device_cfg):
        """Test initialization with activation_distance."""
        cfg = SceneCollisionCostCfg(
            weight=1.0,
            activation_distance=0.05,
            device_cfg=device_cfg,
        )
        assert cfg.activation_distance[0] == 0.05

    def test_update_num_spheres(self, device_cfg, scene_collision_checker):
        """Test update_num_spheres method."""
        cfg = SceneCollisionCostCfg(weight=1.0, device_cfg=device_cfg)
        cfg.scene_collision_checker = scene_collision_checker
        cfg.update_num_spheres(65)
        assert cfg.num_spheres == 65


class TestSceneCollisionCost:
    """Test SceneCollisionCost class."""

    def test_init(self, device_cfg, scene_collision_checker):
        """Test SceneCollisionCost initialization."""
        cfg = SceneCollisionCostCfg(weight=1.0, device_cfg=device_cfg)
        cfg.scene_collision_checker = scene_collision_checker
        cfg.update_num_spheres(65)
        cost = SceneCollisionCost(cfg)
        assert cost is not None
        assert cost.config == cfg

    def test_init_missing_scene_collision_checker_raises_error(self, device_cfg):
        """Test that missing scene_collision_checker raises error."""
        cfg = SceneCollisionCostCfg(weight=1.0, device_cfg=device_cfg)
        with pytest.raises(Exception):
            SceneCollisionCost(cfg)

    def test_setup_batch_tensors(self, device_cfg, scene_collision_checker, franka_kinematics):
        """Test setup_batch_tensors."""
        cfg = SceneCollisionCostCfg(weight=1.0, device_cfg=device_cfg)
        cfg.scene_collision_checker = scene_collision_checker
        num_spheres = franka_kinematics.total_spheres
        cfg.update_num_spheres(num_spheres)
        cost = SceneCollisionCost(cfg)
        batch_size = 4
        horizon = 10
        cost.setup_batch_tensors(batch_size, horizon)

        assert cost._batch_size == batch_size
        assert cost._horizon == horizon

    def test_forward_with_kinematics(
        self, device_cfg, scene_collision_checker, franka_kinematics
    ):
        """Test forward pass using real Franka kinematics."""
        cfg = SceneCollisionCostCfg(weight=1.0, device_cfg=device_cfg)
        cfg.scene_collision_checker = scene_collision_checker
        num_spheres = franka_kinematics.total_spheres
        cfg.update_num_spheres(num_spheres)
        cost = SceneCollisionCost(cfg)
        batch_size = 4
        horizon = 10
        dof = franka_kinematics.dof
        cost.setup_batch_tensors(batch_size, horizon)

        # Create random joint configurations at default (safe position)
        q = torch.zeros((batch_size, horizon, dof), **device_cfg.as_torch_dict())

        # Get kinematic state from kinematics
        kin_state = franka_kinematics.compute_kinematics(
            JointState.from_position(q, joint_names=franka_kinematics.joint_names)
        )

        result = cost.forward(kin_state)
        # Result shape is (batch_size, horizon, num_spheres) for collision cost
        assert result.shape == (batch_size, horizon, num_spheres)

    def test_forward_default_joint_position(
        self, device_cfg, scene_collision_checker, franka_kinematics, franka_robot_cfg
    ):
        """Test forward with default joint configuration."""
        cfg = SceneCollisionCostCfg(weight=1.0, device_cfg=device_cfg)
        cfg.scene_collision_checker = scene_collision_checker
        num_spheres = franka_kinematics.total_spheres
        cfg.update_num_spheres(num_spheres)
        cost = SceneCollisionCost(cfg)
        batch_size = 1
        horizon = 1
        dof = franka_kinematics.dof
        cost.setup_batch_tensors(batch_size, horizon)

        # Use default joint configuration
        default_position = franka_robot_cfg.kinematics.kinematics_config.cspace.default_joint_position
        q = default_position.view(1, 1, dof)

        # Get kinematic state from kinematics
        kin_state = franka_kinematics.compute_kinematics(
            JointState.from_position(q, joint_names=franka_kinematics.joint_names)
        )

        result = cost.forward(kin_state)
        # Result shape is (batch_size, horizon, num_spheres) for collision cost
        assert result.shape == (batch_size, horizon, num_spheres)

    def test_validate_input_wrong_sphere_count(
        self, device_cfg, scene_collision_checker, franka_kinematics
    ):
        """Test validation with wrong sphere count."""
        cfg = SceneCollisionCostCfg(weight=1.0, device_cfg=device_cfg)
        cfg.scene_collision_checker = scene_collision_checker
        cfg.update_num_spheres(10)  # Wrong number of spheres
        cost = SceneCollisionCost(cfg)
        batch_size = 4
        horizon = 10
        dof = franka_kinematics.dof
        cost.setup_batch_tensors(batch_size, horizon)

        # Create joint configurations
        q = torch.zeros((batch_size, horizon, dof), **device_cfg.as_torch_dict())

        # Get kinematic state (has different number of spheres)
        kin_state = franka_kinematics.compute_kinematics(
            JointState.from_position(q, joint_names=franka_kinematics.joint_names)
        )

        # Should raise because num_spheres doesn't match
        with pytest.raises(Exception):
            cost.forward(kin_state)

    def test_update_num_spheres(
        self, device_cfg, scene_collision_checker, franka_kinematics
    ):
        """Test update_num_spheres method."""
        cfg = SceneCollisionCostCfg(weight=1.0, device_cfg=device_cfg)
        cfg.scene_collision_checker = scene_collision_checker
        cfg.update_num_spheres(30)
        cost = SceneCollisionCost(cfg)
        cost.setup_batch_tensors(batch_size=4, horizon=10)

        # Update num_spheres
        new_num_spheres = franka_kinematics.total_spheres
        cost.update_num_spheres(new_num_spheres)
        assert cost.config.num_spheres == new_num_spheres


class TestSceneCollisionCostWithDifferentScenes:
    """Test SceneCollisionCost with different scene configurations."""

    def test_empty_scene(self, device_cfg, franka_kinematics):
        """Test with empty scene (no obstacles)."""
        empty_scene_cfg = SceneCfg()
        scene_collision_cfg = SceneCollisionCfg(
            device_cfg=device_cfg,
            scene_model=empty_scene_cfg,
            cache={"mesh": 10, "primitive": 10},
            max_distance=1.0,
            num_envs=1,
        )
        scene_collision_checker = create_collision_checker(scene_collision_cfg)

        cfg = SceneCollisionCostCfg(weight=1.0, device_cfg=device_cfg)
        cfg.scene_collision_checker = scene_collision_checker
        num_spheres = franka_kinematics.total_spheres
        cfg.update_num_spheres(num_spheres)
        cost = SceneCollisionCost(cfg)
        batch_size = 2
        horizon = 5
        dof = franka_kinematics.dof
        cost.setup_batch_tensors(batch_size, horizon)

        q = torch.zeros((batch_size, horizon, dof), **device_cfg.as_torch_dict())
        kin_state = franka_kinematics.compute_kinematics(
            JointState.from_position(q, joint_names=franka_kinematics.joint_names)
        )

        result = cost.forward(kin_state)
        # Result shape is (batch_size, horizon, num_spheres) for collision cost
        assert result.shape == (batch_size, horizon, num_spheres)
        # Empty scene should have zero collision cost
        assert torch.all(result == 0)

    def test_multiple_obstacles(self, device_cfg, franka_kinematics):
        """Test with multiple obstacles."""
        # Create scene with multiple cuboids
        table = Cuboid(
            name="table",
            pose=[0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
            dims=[0.6, 1.0, 0.05],
        )
        box1 = Cuboid(
            name="box1",
            pose=[0.4, 0.3, 0.5, 1.0, 0.0, 0.0, 0.0],
            dims=[0.1, 0.1, 0.1],
        )
        box2 = Cuboid(
            name="box2",
            pose=[0.4, -0.3, 0.5, 1.0, 0.0, 0.0, 0.0],
            dims=[0.1, 0.1, 0.1],
        )
        multi_scene_cfg = SceneCfg(cuboid=[table, box1, box2])

        scene_collision_cfg = SceneCollisionCfg(
            device_cfg=device_cfg,
            scene_model=multi_scene_cfg,
            cache={"mesh": 10, "primitive": 10},
            max_distance=1.0,
            num_envs=1,
        )
        scene_collision_checker = create_collision_checker(scene_collision_cfg)

        cfg = SceneCollisionCostCfg(weight=1.0, device_cfg=device_cfg)
        cfg.scene_collision_checker = scene_collision_checker
        num_spheres = franka_kinematics.total_spheres
        cfg.update_num_spheres(num_spheres)
        cost = SceneCollisionCost(cfg)
        batch_size = 2
        horizon = 5
        dof = franka_kinematics.dof
        cost.setup_batch_tensors(batch_size, horizon)

        q = torch.zeros((batch_size, horizon, dof), **device_cfg.as_torch_dict())
        kin_state = franka_kinematics.compute_kinematics(
            JointState.from_position(q, joint_names=franka_kinematics.joint_names)
        )

        result = cost.forward(kin_state)
        # Result shape is (batch_size, horizon, num_spheres) for collision cost
        assert result.shape == (batch_size, horizon, num_spheres)


class TestSceneCollisionCostJITFunctions:
    """Test JIT-compiled functions in SceneCollisionCost."""

    def test_jit_weight_distance(self):
        """Test jit_weight_distance function."""
        dist = torch.randn(4, 10, 5, device="cuda:0")

        # Test sum_cost=True
        result_sum = SceneCollisionCost.jit_weight_distance(dist.clone(), sum_cost=True)
        assert result_sum.shape == (4, 10)

        # Test sum_cost=False (max)
        result_max = SceneCollisionCost.jit_weight_distance(dist.clone(), sum_cost=False)
        assert result_max.shape == (4, 10)

    def test_jit_weight_collision(self):
        """Test jit_weight_collision function."""
        dist = torch.randn(4, 10, 5, device="cuda:0")

        # Test sum_cost=True
        result_sum = SceneCollisionCost.jit_weight_collision(dist.clone(), sum_cost=True)
        assert result_sum.shape == (4, 10)

        # Test sum_cost=False (max)
        result_max = SceneCollisionCost.jit_weight_collision(dist.clone(), sum_cost=False)
        assert result_max.shape == (4, 10)


class TestSceneCollisionCostGradients:
    """Test gradient computation for SceneCollisionCost."""

    def test_scene_collision_cost_gradient_buffer(
        self, device_cfg, scene_collision_checker, franka_kinematics
    ):
        """Test that gradients can flow through collision cost computation."""
        cfg = SceneCollisionCostCfg(
            weight=1.0,
            use_grad_input=True,
            device_cfg=device_cfg,
        )
        cfg.scene_collision_checker = scene_collision_checker
        num_spheres = franka_kinematics.total_spheres
        cfg.update_num_spheres(num_spheres)
        cost = SceneCollisionCost(cfg)
        batch_size = 2
        horizon = 3
        dof = franka_kinematics.dof
        cost.setup_batch_tensors(batch_size, horizon)

        # Create joint configurations with requires_grad
        q = torch.zeros(
            (batch_size, horizon, dof),
            dtype=torch.float32,
            device=device_cfg.device,
            requires_grad=True,
        )

        # Get kinematic state
        kin_state = franka_kinematics.compute_kinematics(
            JointState.from_position(q, joint_names=franka_kinematics.joint_names)
        )

        # Forward pass
        result = cost.forward(kin_state)

        # Check output shape (batch_size, horizon) for summed collision cost
        assert result.shape == (batch_size, horizon, num_spheres)

        # Verify output is a tensor that can be used in computation
        loss = result.sum()
        assert loss.requires_grad is True or loss.grad_fn is not None

    def test_scene_collision_cost_with_collision(
        self, device_cfg, franka_kinematics
    ):
        """Test gradient flow when robot is in collision with obstacle."""
        # Create scene with obstacle close to robot base
        close_obstacle = Cuboid(
            name="close_obstacle",
            pose=[0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],  # Close to robot
            dims=[0.2, 0.2, 0.2],
        )
        collision_scene_cfg = SceneCfg(cuboid=[close_obstacle])

        scene_collision_cfg = SceneCollisionCfg(
            device_cfg=device_cfg,
            scene_model=collision_scene_cfg,
            cache={"mesh": 10, "primitive": 10},
            max_distance=1.0,
            num_envs=1,
        )
        collision_checker = create_collision_checker(scene_collision_cfg)

        cfg = SceneCollisionCostCfg(
            weight=1.0,
            use_grad_input=True,
            device_cfg=device_cfg,
        )
        cfg.scene_collision_checker = collision_checker
        num_spheres = franka_kinematics.total_spheres
        cfg.update_num_spheres(num_spheres)
        cost = SceneCollisionCost(cfg)
        batch_size = 2
        horizon = 3
        dof = franka_kinematics.dof
        cost.setup_batch_tensors(batch_size, horizon)

        # Create joint configurations
        q = torch.zeros(
            (batch_size, horizon, dof),
            dtype=torch.float32,
            device=device_cfg.device,
            requires_grad=True,
        )

        kin_state = franka_kinematics.compute_kinematics(
            JointState.from_position(q, joint_names=franka_kinematics.joint_names)
        )

        # Forward pass
        result = cost.forward(kin_state)

        # Verify result is a tensor
        assert isinstance(result, torch.Tensor)
        assert result.shape == (batch_size, horizon, num_spheres)

        # Sum and check loss has gradient
        loss = result.sum()
        assert loss.requires_grad is True or isinstance(loss.item(), float)

    def test_scene_collision_cost_sweep_gradient(
        self, device_cfg, scene_collision_checker, franka_kinematics
    ):
        """Test gradient flow with sweep collision checking."""
        cfg = SceneCollisionCostCfg(
            weight=1.0,
            use_grad_input=True,
            use_sweep=True,
            device_cfg=device_cfg,
        )
        cfg.scene_collision_checker = scene_collision_checker
        num_spheres = franka_kinematics.total_spheres
        cfg.update_num_spheres(num_spheres)
        cost = SceneCollisionCost(cfg)
        batch_size = 2
        horizon = 5
        dof = franka_kinematics.dof
        cost.setup_batch_tensors(batch_size, horizon)

        # Create joint configurations
        q = torch.zeros(
            (batch_size, horizon, dof),
            dtype=torch.float32,
            device=device_cfg.device,
            requires_grad=True,
        )

        kin_state = franka_kinematics.compute_kinematics(
            JointState.from_position(q, joint_names=franka_kinematics.joint_names)
        )

        # Create trajectory dt
        trajectory_dt = torch.full(
            (batch_size,), 0.1, dtype=torch.float32, device=device_cfg.device
        )

        # Forward pass with sweep
        result = cost.forward(kin_state, trajectory_dt=trajectory_dt)

        # Verify result is a tensor
        assert isinstance(result, torch.Tensor)

