# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for GoalRegistry.

Tests the goal_registry.py module which manages goal specifications,
indexing for batch/seed optimization, and handles different goal types
(joint state goals, link pose goals, multi-link goalset poses).
"""

# Standard Library

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.state.state_joint import JointState
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose


@pytest.fixture
def sample_joint_state(cuda_device_cfg):
    """Create a sample joint state for testing."""
    batch_size = 4
    dof = 7
    position = torch.randn(batch_size, dof, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
    velocity = torch.randn(batch_size, dof, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
    acceleration = torch.randn(
        batch_size, dof, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
    )
    return JointState(
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        device_cfg=cuda_device_cfg,
    )


@pytest.fixture
def sample_seed_joint_state(cuda_device_cfg):
    """Create a sample seed joint state for testing."""
    batch_size = 2
    num_seeds = 3
    dof = 7
    position = torch.randn(
        batch_size, num_seeds, dof, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
    )
    velocity = torch.randn(
        batch_size, num_seeds, dof, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
    )
    acceleration = torch.randn(
        batch_size, num_seeds, dof, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
    )
    return JointState(
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        device_cfg=cuda_device_cfg,
    )


@pytest.fixture
def sample_pose(cuda_device_cfg):
    """Create a sample 2D pose for testing."""
    batch_size = 4
    position = torch.randn(
        batch_size, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
    )
    quaternion = torch.randn(
        batch_size, 4, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
    )
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)
    return Pose(position=position, quaternion=quaternion, name="ee_link")


@pytest.fixture
def sample_multi_link_goalset_pose(cuda_device_cfg):
    """Create a sample multi-link goalset GoalToolPose for testing."""
    batch_size = 4
    num_goalset = 2
    num_links = 2
    tool_frames = ["link1", "link2"]

    # 5D: [batch, horizon=1, num_links, num_goalset, 3/4]
    positions = torch.randn(
        batch_size, 1, num_links, num_goalset, 3,
        device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype,
    )
    quaternions = torch.randn(
        batch_size, 1, num_links, num_goalset, 4,
        device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype,
    )
    quaternions = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)

    return GoalToolPose(tool_frames=tool_frames, position=positions, quaternion=quaternions)


class TestGoalRegistryInitialization:
    """Test GoalRegistry initialization."""

    def test_default_initialization(self):
        """Test default initialization without parameters."""
        registry = GoalRegistry()

        assert registry.name == "goal"
        assert registry.batch_size == -1
        assert registry.num_goalset == 1
        assert registry.num_seeds == 1
        assert registry.goal_js is None
        assert registry.seed_goal_js is None
        assert registry.link_goal_poses is None
        assert registry.current_js is None

    def test_initialization_with_joint_state(self, sample_joint_state):
        """Test initialization with a joint state goal."""
        registry = GoalRegistry(goal_js=sample_joint_state)

        assert registry.goal_js is not None
        assert registry.batch_size == sample_joint_state.position.shape[0]

    def test_initialization_with_link_goal_poses(self, sample_multi_link_goalset_pose):
        """Test initialization with link goal poses."""
        registry = GoalRegistry(link_goal_poses=sample_multi_link_goalset_pose)

        assert registry.link_goal_poses is not None
        assert registry.batch_size == sample_multi_link_goalset_pose.batch_size
        assert registry.num_goalset == sample_multi_link_goalset_pose.num_goalset
        assert registry.idxs_link_pose is not None
        assert registry.idxs_link_pose.shape[0] == sample_multi_link_goalset_pose.batch_size

    def test_initialization_with_current_js(self, sample_joint_state):
        """Test initialization with current joint state."""
        registry = GoalRegistry(current_js=sample_joint_state)

        assert registry.current_js is not None
        assert registry.idxs_current_js is not None
        assert registry.idxs_current_js.shape[0] == sample_joint_state.position.shape[0]

    def test_initialization_with_seed_goal_js(self, sample_seed_joint_state):
        """Test initialization with seed goal joint state."""
        registry = GoalRegistry(seed_goal_js=sample_seed_joint_state)

        assert registry.seed_goal_js is not None
        assert registry.idxs_seed_goal_js is not None
        assert registry.seed_enable_implicit_goal_js is not None

        batch_size, num_seeds, _ = sample_seed_joint_state.position.shape
        expected_idx_size = batch_size * num_seeds
        assert registry.idxs_seed_goal_js.shape[0] == expected_idx_size
        assert registry.seed_enable_implicit_goal_js.shape == (batch_size, num_seeds)


class TestGoalRegistryIndexing:
    """Test GoalRegistry indexing functionality."""

    def test_idxs_link_pose_automatic_creation(self, sample_multi_link_goalset_pose):
        """Test that idxs_link_pose is automatically created."""
        registry = GoalRegistry(link_goal_poses=sample_multi_link_goalset_pose)

        assert registry.idxs_link_pose is not None
        assert registry.idxs_link_pose.dtype == torch.int32
        assert registry.idxs_link_pose.ndim == 2
        assert registry.idxs_link_pose.shape[1] == 1

    def test_idxs_current_js_automatic_creation(self, sample_joint_state):
        """Test that idxs_current_js is automatically created."""
        registry = GoalRegistry(current_js=sample_joint_state)

        assert registry.idxs_current_js is not None
        assert registry.idxs_current_js.dtype == torch.int32
        expected_indices = torch.arange(
            sample_joint_state.position.shape[0],
            device=sample_joint_state.position.device,
            dtype=torch.int32
        )
        assert torch.all(registry.idxs_current_js[:, 0] == expected_indices)

    def test_idxs_seed_goal_js_automatic_creation(self, sample_seed_joint_state):
        """Test that idxs_seed_goal_js is automatically created."""
        registry = GoalRegistry(seed_goal_js=sample_seed_joint_state)

        batch_size, num_seeds, _ = sample_seed_joint_state.position.shape
        expected_size = batch_size * num_seeds

        assert registry.idxs_seed_goal_js is not None
        assert registry.idxs_seed_goal_js.dtype == torch.int32
        assert registry.idxs_seed_goal_js.shape[0] == expected_size

    def test_custom_idxs_preserved(self, sample_multi_link_goalset_pose, cuda_device_cfg):
        """Test that custom indices are preserved if provided."""
        batch_size = sample_multi_link_goalset_pose.batch_size
        custom_idxs = torch.zeros(
            batch_size, 1, device=cuda_device_cfg.device, dtype=torch.int32
        )

        registry = GoalRegistry(
            link_goal_poses=sample_multi_link_goalset_pose, idxs_link_pose=custom_idxs
        )

        assert torch.all(registry.idxs_link_pose == custom_idxs)


class TestGoalRegistryProperties:
    """Test GoalRegistry properties."""

    def test_link_goal_pose_dict_with_no_poses(self):
        """Test link_goal_pose_dict returns None when no poses."""
        registry = GoalRegistry()

        assert registry.link_goal_pose_dict is None

    def test_link_goal_pose_dict_with_poses(self, sample_multi_link_goalset_pose):
        """Test link_goal_pose_dict returns dict when poses exist."""
        registry = GoalRegistry(link_goal_poses=sample_multi_link_goalset_pose)

        pose_dict = registry.link_goal_pose_dict

        assert pose_dict is not None
        assert isinstance(pose_dict, dict)
        assert set(pose_dict.keys()) == set(sample_multi_link_goalset_pose.tool_frames)

    def test_get_index_size_with_link_pose(self, sample_multi_link_goalset_pose):
        """Test get_index_size with link pose indices."""
        registry = GoalRegistry(link_goal_poses=sample_multi_link_goalset_pose)

        index_size = registry.get_index_size()

        assert index_size == sample_multi_link_goalset_pose.batch_size

    def test_get_index_size_with_goal_js(self, sample_joint_state):
        """Test get_index_size with goal joint state indices."""
        batch_size = sample_joint_state.position.shape[0]
        idxs = torch.arange(batch_size, dtype=torch.int32).unsqueeze(-1)
        registry = GoalRegistry(goal_js=sample_joint_state, idxs_goal_js=idxs)

        index_size = registry.get_index_size()

        assert index_size == batch_size

    def test_get_index_size_with_no_indices(self):
        """Test get_index_size returns None when no indices."""
        registry = GoalRegistry()

        index_size = registry.get_index_size()

        assert index_size is None


class TestGoalRegistryClone:
    """Test GoalRegistry clone functionality."""

    def test_clone_empty_registry(self):
        """Test cloning an empty registry."""
        registry = GoalRegistry()
        cloned = registry.clone()

        assert cloned is not registry
        assert cloned.name == registry.name
        assert cloned.batch_size == registry.batch_size
        assert cloned.num_goalset == registry.num_goalset
        assert cloned.num_seeds == registry.num_seeds

    def test_clone_with_joint_state(self, sample_joint_state):
        """Test cloning with joint state."""
        registry = GoalRegistry(goal_js=sample_joint_state)
        cloned = registry.clone()

        assert cloned is not registry
        assert cloned.goal_js is registry.goal_js  # Shallow copy
        assert cloned.batch_size == registry.batch_size

    def test_clone_with_link_goal_poses(self, sample_multi_link_goalset_pose):
        """Test cloning with link goal poses."""
        registry = GoalRegistry(link_goal_poses=sample_multi_link_goalset_pose)
        cloned = registry.clone()

        assert cloned is not registry
        assert cloned.link_goal_poses is not registry.link_goal_poses  # Deep clone
        assert cloned.num_goalset == registry.num_goalset

    def test_clone_with_seed_goal_js(self, sample_seed_joint_state):
        """Test cloning with seed goal joint state."""
        registry = GoalRegistry(seed_goal_js=sample_seed_joint_state)
        cloned = registry.clone()

        assert cloned is not registry
        assert cloned.seed_goal_js is registry.seed_goal_js
        assert cloned.idxs_seed_goal_js is registry.idxs_seed_goal_js

    def test_clone_preserves_all_fields(self, sample_joint_state, cuda_device_cfg):
        """Test that clone preserves all fields."""
        registry = GoalRegistry(
            goal_js=sample_joint_state,
            current_js=sample_joint_state,
            num_seeds=2,
        )
        cloned = registry.clone()

        assert cloned.goal_js is registry.goal_js
        assert cloned.current_js is registry.current_js
        assert cloned.num_seeds == registry.num_seeds


class TestGoalRegistryRepeatSeeds:
    """Test GoalRegistry repeat_seeds functionality."""

    def test_repeat_seeds_basic(self, sample_joint_state):
        """Test basic seed repetition."""
        registry = GoalRegistry(goal_js=sample_joint_state)
        num_seeds = 4

        repeated = registry.repeat_seeds(num_seeds)

        assert repeated is not registry
        assert repeated.num_seeds == num_seeds * registry.num_seeds
        assert repeated.batch_size == registry.batch_size

    def test_repeat_seeds_with_link_poses(self, sample_multi_link_goalset_pose):
        """Test seed repetition with link goal poses."""
        registry = GoalRegistry(link_goal_poses=sample_multi_link_goalset_pose)
        num_seeds = 3

        repeated = registry.repeat_seeds(num_seeds)

        assert repeated.link_goal_poses is registry.link_goal_poses
        assert repeated.num_goalset == registry.num_goalset
        assert repeated.idxs_link_pose.shape[0] == registry.idxs_link_pose.shape[0] * num_seeds

    def test_repeat_seeds_with_current_js(self, sample_joint_state):
        """Test seed repetition with current joint state."""
        registry = GoalRegistry(current_js=sample_joint_state)
        num_seeds = 2

        repeated = registry.repeat_seeds(num_seeds)

        assert repeated.current_js is not None
        assert repeated.idxs_current_js.shape[0] == registry.idxs_current_js.shape[0] * num_seeds

    def test_repeat_seeds_with_seed_goal_js(self, sample_seed_joint_state):
        """Test seed repetition with seed goal joint state."""
        registry = GoalRegistry(seed_goal_js=sample_seed_joint_state)
        num_seeds = 2

        repeated = registry.repeat_seeds(num_seeds)

        assert repeated.seed_goal_js is not None
        # idxs_seed_goal_js should not be repeated by default
        assert repeated.idxs_seed_goal_js.shape == registry.idxs_seed_goal_js.shape

    def test_repeat_seeds_with_seed_idx_buffers(self, sample_seed_joint_state):
        """Test seed repetition with repeat_seed_idx_buffers=True."""
        registry = GoalRegistry(seed_goal_js=sample_seed_joint_state)
        num_seeds = 2

        repeated = registry.repeat_seeds(num_seeds, repeat_seed_idx_buffers=True)

        assert repeated.seed_goal_js is not None
        assert repeated.idxs_seed_goal_js.shape[0] == registry.idxs_seed_goal_js.shape[0] * num_seeds

    def test_repeat_seeds_preserves_enable_indices(self, sample_multi_link_goalset_pose, cuda_device_cfg):
        """Test that repeat_seeds preserves enable indices."""
        batch_size = sample_multi_link_goalset_pose.batch_size
        enable_idxs = torch.ones(batch_size, 2, device=cuda_device_cfg.device, dtype=torch.int32)

        registry = GoalRegistry(
            link_goal_poses=sample_multi_link_goalset_pose, idxs_enable=enable_idxs
        )
        num_seeds = 3

        repeated = registry.repeat_seeds(num_seeds)

        assert repeated.idxs_enable is not None
        assert repeated.idxs_enable.shape[0] == enable_idxs.shape[0] * num_seeds

    def test_repeat_seeds_preserves_env_indices(self, sample_multi_link_goalset_pose, cuda_device_cfg):
        """Test that repeat_seeds preserves env indices."""
        batch_size = sample_multi_link_goalset_pose.batch_size
        env_idxs = torch.zeros(batch_size, 1, device=cuda_device_cfg.device, dtype=torch.int32)

        registry = GoalRegistry(
            link_goal_poses=sample_multi_link_goalset_pose, idxs_env=env_idxs
        )
        num_seeds = 2

        repeated = registry.repeat_seeds(num_seeds)

        assert repeated.idxs_env is not None
        assert repeated.idxs_env.shape[0] == env_idxs.shape[0] * num_seeds


class TestGoalRegistryCopy:
    """Test GoalRegistry copy_ functionality."""

    def test_copy_empty_to_empty(self):
        """Test copying empty registry to empty registry."""
        source = GoalRegistry()
        target = GoalRegistry()

        target.copy_(source)

        assert target.batch_size == source.batch_size

    def test_copy_goal_js(self, sample_joint_state):
        """Test copying goal joint state."""
        source = GoalRegistry(goal_js=sample_joint_state)
        target = GoalRegistry(goal_js=sample_joint_state.clone())

        original_position = target.goal_js.position.clone()
        source.goal_js.position[:] = 1.0

        target.copy_(source)

        assert torch.all(target.goal_js.position == 1.0)
        assert not torch.allclose(target.goal_js.position, original_position)

    def test_copy_current_js(self, sample_joint_state):
        """Test copying current joint state."""
        source = GoalRegistry(current_js=sample_joint_state)
        target = GoalRegistry(current_js=sample_joint_state.clone())

        original_position = target.current_js.position.clone()
        source.current_js.position[:] = 2.0

        target.copy_(source)

        assert torch.all(target.current_js.position == 2.0)
        assert not torch.allclose(target.current_js.position, original_position)

    def test_copy_seed_goal_js(self, sample_seed_joint_state):
        """Test copying seed goal joint state."""
        source = GoalRegistry(seed_goal_js=sample_seed_joint_state)
        target = GoalRegistry(seed_goal_js=sample_seed_joint_state.clone())

        original_position = target.seed_goal_js.position.clone()
        source.seed_goal_js.position[:] = 3.0

        target.copy_(source)

        assert torch.all(target.seed_goal_js.position == 3.0)
        assert not torch.allclose(target.seed_goal_js.position, original_position)

    def test_copy_link_goal_poses(self, sample_multi_link_goalset_pose):
        """Test copying link goal poses."""
        source = GoalRegistry(link_goal_poses=sample_multi_link_goalset_pose)
        target = GoalRegistry(link_goal_poses=sample_multi_link_goalset_pose.clone())

        original_position = target.link_goal_poses.position.clone()
        source.link_goal_poses.position[:] = 5.0

        target.copy_(source)

        assert torch.all(target.link_goal_poses.position == 5.0)
        assert not torch.allclose(target.link_goal_poses.position, original_position)

    def test_copy_with_allow_clone(self, sample_joint_state):
        """Test copy with allow_clone=True creates new objects."""
        source = GoalRegistry(goal_js=sample_joint_state)
        target = GoalRegistry()

        target.copy_(source, allow_clone=True)

        assert target.goal_js is source.goal_js

    def test_copy_without_allow_clone(self, sample_joint_state):
        """Test copy with allow_clone=False doesn't clone."""
        source = GoalRegistry(goal_js=sample_joint_state)
        target = GoalRegistry()

        target.copy_(source, allow_clone=False)

        # Should not have created goal_js since target didn't have it
        assert target.goal_js is None

    def test_copy_updates_batch_size(self, sample_joint_state):
        """Test that copy updates batch size."""
        source = GoalRegistry(goal_js=sample_joint_state)
        target = GoalRegistry()

        target.copy_(source, allow_clone=True)

        assert target.batch_size == source.batch_size

    def test_copy_with_update_idx_buffers(self, sample_multi_link_goalset_pose, cuda_device_cfg):
        """Test copy with update_idx_buffers=True."""
        batch_size = sample_multi_link_goalset_pose.batch_size
        custom_idxs = torch.ones(batch_size, 1, device=cuda_device_cfg.device, dtype=torch.int32)

        source = GoalRegistry(
            link_goal_poses=sample_multi_link_goalset_pose, idxs_link_pose=custom_idxs
        )
        source.update_idxs_buffers = True

        target = GoalRegistry(link_goal_poses=sample_multi_link_goalset_pose.clone())

        target.copy_(source, update_idx_buffers=True)

        assert torch.all(target.idxs_link_pose == custom_idxs)

    def test_copy_without_update_idx_buffers(self, sample_multi_link_goalset_pose, cuda_device_cfg):
        """Test copy with update_idx_buffers=False."""
        batch_size = sample_multi_link_goalset_pose.batch_size
        custom_idxs = torch.ones(batch_size, 1, device=cuda_device_cfg.device, dtype=torch.int32)

        source = GoalRegistry(
            link_goal_poses=sample_multi_link_goalset_pose, idxs_link_pose=custom_idxs
        )
        source.update_idxs_buffers = True

        target = GoalRegistry(link_goal_poses=sample_multi_link_goalset_pose.clone())
        original_idxs = target.idxs_link_pose.clone()

        target.copy_(source, update_idx_buffers=False)

        # Indices should not have been updated
        assert torch.all(target.idxs_link_pose == original_idxs)


class TestGoalRegistryApplyKernel:
    """Test GoalRegistry apply_kernel functionality."""

    def test_apply_kernel_with_link_poses(self, sample_multi_link_goalset_pose, cuda_device_cfg):
        """Test applying kernel matrix to link poses."""
        batch_size = sample_multi_link_goalset_pose.batch_size
        new_batch = batch_size * 2

        # Create identity-like kernel that doubles batch
        kernel_mat = torch.zeros(
            new_batch, batch_size, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
        )
        for i in range(batch_size):
            kernel_mat[i, i] = 1.0
            kernel_mat[i + batch_size, i] = 1.0

        registry = GoalRegistry(link_goal_poses=sample_multi_link_goalset_pose)
        transformed = registry.apply_kernel(kernel_mat)

        assert transformed.link_goal_poses is registry.link_goal_poses
        assert transformed.idxs_link_pose is not None
        assert transformed.idxs_link_pose.shape[0] == new_batch

    def test_apply_kernel_with_goal_js(self, sample_joint_state, cuda_device_cfg):
        """Test applying kernel matrix with goal joint state."""
        batch_size = sample_joint_state.position.shape[0]
        new_batch = batch_size * 2

        kernel_mat = torch.zeros(
            new_batch, batch_size, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
        )
        for i in range(batch_size):
            kernel_mat[i, i] = 1.0
            kernel_mat[i + batch_size, i] = 1.0

        idxs = torch.arange(batch_size, dtype=torch.int32, device=cuda_device_cfg.device).unsqueeze(-1)
        registry = GoalRegistry(goal_js=sample_joint_state, idxs_goal_js=idxs)

        transformed = registry.apply_kernel(kernel_mat)

        assert transformed.goal_js is registry.goal_js
        assert transformed.idxs_goal_js is not None
        assert transformed.idxs_goal_js.shape[0] == new_batch

    def test_apply_kernel_with_current_js(self, sample_joint_state, cuda_device_cfg):
        """Test applying kernel matrix with current joint state."""
        batch_size = sample_joint_state.position.shape[0]
        new_batch = batch_size

        kernel_mat = torch.eye(
            new_batch, batch_size, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
        )

        registry = GoalRegistry(current_js=sample_joint_state)
        transformed = registry.apply_kernel(kernel_mat)

        assert transformed.current_js is registry.current_js
        assert transformed.idxs_current_js is not None

    def test_apply_kernel_with_enable_indices(self, sample_multi_link_goalset_pose, cuda_device_cfg):
        """Test applying kernel matrix with enable indices."""
        batch_size = sample_multi_link_goalset_pose.batch_size
        enable_idxs = torch.ones(batch_size, 2, device=cuda_device_cfg.device)

        kernel_mat = torch.eye(
            batch_size, batch_size, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
        )

        registry = GoalRegistry(
            link_goal_poses=sample_multi_link_goalset_pose, idxs_enable=enable_idxs
        )
        transformed = registry.apply_kernel(kernel_mat)

        assert transformed.idxs_enable is not None
        assert transformed.idxs_enable.shape[0] == batch_size

    def test_apply_kernel_with_env_indices(self, sample_multi_link_goalset_pose, cuda_device_cfg):
        """Test applying kernel matrix with env indices."""
        batch_size = sample_multi_link_goalset_pose.batch_size
        env_idxs = torch.zeros(
            batch_size, 1, device=cuda_device_cfg.device, dtype=torch.int32
        )

        kernel_mat = torch.eye(
            batch_size, batch_size, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
        )

        registry = GoalRegistry(
            link_goal_poses=sample_multi_link_goalset_pose, idxs_env=env_idxs
        )
        transformed = registry.apply_kernel(kernel_mat)

        assert transformed.idxs_env is not None
        assert transformed.idxs_env.dtype == torch.int32


class TestGoalRegistryCreateIdx:
    """Test GoalRegistry create_idx class method."""

    def test_create_idx_basic(self, cuda_device_cfg):
        """Test basic create_idx."""
        batch_size = 8
        num_seeds = 2

        registry = GoalRegistry.create_idx(
            pose_batch_size=batch_size,
            multi_env=False,
            num_seeds=num_seeds,
            device_cfg=cuda_device_cfg,
        )

        expected_total_batch = batch_size * num_seeds

        assert registry.idxs_link_pose is not None
        assert registry.idxs_link_pose.shape[0] == expected_total_batch
        assert registry.idxs_env is not None
        assert registry.idxs_current_js is not None
        assert registry.idxs_goal_js is not None

    def test_create_idx_with_multi_env(self, cuda_device_cfg):
        """Test create_idx with multi_env=True."""
        batch_size = 4
        num_seeds = 2

        registry = GoalRegistry.create_idx(
            pose_batch_size=batch_size,
            multi_env=True,
            num_seeds=num_seeds,
            device_cfg=cuda_device_cfg,
        )

        # Env indices should match pose indices when multi_env=True
        expected_total_batch = batch_size * num_seeds
        assert registry.idxs_env.shape[0] == expected_total_batch

        # Check that env indices are not all zeros
        assert not torch.all(registry.idxs_env == 0)

    def test_create_idx_without_multi_env(self, cuda_device_cfg):
        """Test create_idx with multi_env=False."""
        batch_size = 4
        num_seeds = 2

        registry = GoalRegistry.create_idx(
            pose_batch_size=batch_size,
            multi_env=False,
            num_seeds=num_seeds,
            device_cfg=cuda_device_cfg,
        )

        # Env indices should be all zeros when multi_env=False
        assert torch.all(registry.idxs_env == 0)

    def test_create_idx_with_seed_goal_state(self, sample_seed_joint_state, cuda_device_cfg):
        """Test create_idx with seed goal state."""
        batch_size = sample_seed_joint_state.position.shape[0]
        num_seeds = 2

        registry = GoalRegistry.create_idx(
            pose_batch_size=batch_size,
            multi_env=False,
            num_seeds=num_seeds,
            device_cfg=cuda_device_cfg,
            seed_goal_state=sample_seed_joint_state,
        )

        # repeat_seeds clones the seed_goal_js, so it's not the same object
        assert registry.seed_goal_js is not None
        assert registry.idxs_seed_goal_js is not None
        assert torch.allclose(registry.seed_goal_js.position, sample_seed_joint_state.position)

    def test_create_idx_with_repeat_seed_idx_buffers(self, cuda_device_cfg):
        """Test create_idx with repeat_seed_idx_buffers=True."""
        batch_size = 4
        num_seeds = 3

        registry = GoalRegistry.create_idx(
            pose_batch_size=batch_size,
            multi_env=False,
            num_seeds=num_seeds,
            device_cfg=cuda_device_cfg,
            repeat_seed_idx_buffers=True,
        )

        expected_total_batch = batch_size * num_seeds
        assert registry.idxs_link_pose.shape[0] == expected_total_batch

    def test_create_idx_dtype_consistency(self, cuda_device_cfg):
        """Test that create_idx creates indices with correct dtype."""
        batch_size = 4
        num_seeds = 2

        registry = GoalRegistry.create_idx(
            pose_batch_size=batch_size,
            multi_env=True,
            num_seeds=num_seeds,
            device_cfg=cuda_device_cfg,
        )

        assert registry.idxs_link_pose.dtype == torch.int32
        assert registry.idxs_env.dtype == torch.int32
        assert registry.idxs_current_js.dtype == torch.int32
        assert registry.idxs_goal_js.dtype == torch.int32

    def test_create_idx_device_consistency(self, cuda_device_cfg):
        """Test that create_idx creates indices on correct device."""
        batch_size = 4
        num_seeds = 2

        registry = GoalRegistry.create_idx(
            pose_batch_size=batch_size,
            multi_env=True,
            num_seeds=num_seeds,
            device_cfg=cuda_device_cfg,
        )

        assert registry.idxs_link_pose.device == cuda_device_cfg.device
        assert registry.idxs_env.device == cuda_device_cfg.device
        assert registry.idxs_current_js.device == cuda_device_cfg.device
        assert registry.idxs_goal_js.device == cuda_device_cfg.device


class TestGoalRegistryCreateIndexBuffers:
    """Test GoalRegistry create_index_buffers method."""

    def test_create_index_buffers_basic(self, sample_joint_state, cuda_device_cfg):
        """Test basic create_index_buffers."""
        batch_size = 8
        num_seeds = 2

        source_registry = GoalRegistry(goal_js=sample_joint_state)

        new_registry = source_registry.create_index_buffers(
            batch_size=batch_size,
            multi_env=False,
            num_seeds=num_seeds,
            device_cfg=cuda_device_cfg,
        )

        assert new_registry.goal_js is source_registry.goal_js
        assert new_registry.idxs_link_pose is not None

    def test_create_index_buffers_preserves_goals(self, sample_multi_link_goalset_pose, cuda_device_cfg):
        """Test that create_index_buffers preserves goal data."""
        batch_size = 8
        num_seeds = 2

        source_registry = GoalRegistry(link_goal_poses=sample_multi_link_goalset_pose)

        new_registry = source_registry.create_index_buffers(
            batch_size=batch_size,
            multi_env=True,
            num_seeds=num_seeds,
            device_cfg=cuda_device_cfg,
        )

        assert new_registry.link_goal_poses is source_registry.link_goal_poses


class TestGoalRegistryGetBatchGoalState:
    """Test GoalRegistry get_batch_goal_state method."""

    def test_get_batch_goal_state(self, sample_joint_state):
        """Test get_batch_goal_state retrieves indexed goals."""
        batch_size = sample_joint_state.position.shape[0]
        idxs = torch.arange(batch_size, dtype=torch.int32).unsqueeze(-1)

        registry = GoalRegistry(goal_js=sample_joint_state, idxs_link_pose=idxs)

        batch_goals = registry.get_batch_goal_state()

        assert batch_goals is not None
        assert batch_goals.position.shape[0] == batch_size


class TestGoalRegistryEdgeCases:
    """Test GoalRegistry edge cases."""

    def test_empty_registry_operations(self):
        """Test that operations on empty registry don't crash."""
        registry = GoalRegistry()

        cloned = registry.clone()
        assert cloned is not registry

        repeated = registry.repeat_seeds(2)
        assert repeated is not registry

    def test_large_batch_size(self, cuda_device_cfg):
        """Test with large batch size."""
        batch_size = 1024
        num_seeds = 4

        registry = GoalRegistry.create_idx(
            pose_batch_size=batch_size,
            multi_env=True,
            num_seeds=num_seeds,
            device_cfg=cuda_device_cfg,
        )

        assert registry.idxs_link_pose.shape[0] == batch_size * num_seeds

    def test_single_batch(self, cuda_device_cfg):
        """Test with single batch size."""
        batch_size = 1
        dof = 7
        position = torch.randn(batch_size, dof, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        joint_state = JointState(position=position, device_cfg=cuda_device_cfg)

        registry = GoalRegistry(goal_js=joint_state)

        assert registry.batch_size == batch_size
        assert registry.idxs_goal_js is None  # Not automatically created for goal_js

    def test_many_goalsets(self, cuda_device_cfg):
        """Test with many goalsets."""
        batch_size = 4
        num_goalset = 10
        num_links = 3
        tool_frames = [f"link{i}" for i in range(num_links)]

        # 5D: [batch, horizon=1, num_links, num_goalset, 3/4]
        positions = torch.randn(
            batch_size, 1, num_links, num_goalset, 3,
            device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype,
        )
        quaternions = torch.randn(
            batch_size, 1, num_links, num_goalset, 4,
            device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype,
        )
        quaternions = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)

        multi_link_pose = GoalToolPose(
            tool_frames=tool_frames, position=positions, quaternion=quaternions,
        )

        registry = GoalRegistry(link_goal_poses=multi_link_pose)

        assert registry.num_goalset == num_goalset

    def test_mixed_goal_types(self, sample_joint_state, sample_multi_link_goalset_pose):
        """Test registry with multiple goal types."""
        registry = GoalRegistry(
            goal_js=sample_joint_state,
            link_goal_poses=sample_multi_link_goalset_pose,
            current_js=sample_joint_state,
        )

        assert registry.goal_js is not None
        assert registry.link_goal_poses is not None
        assert registry.current_js is not None
        # Batch size should come from link_goal_poses
        assert registry.batch_size == sample_multi_link_goalset_pose.batch_size
