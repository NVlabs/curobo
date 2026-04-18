# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for GoalManager class."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.solver.manager_goal import GoalManager
from curobo._src.solver.solve_mode import SolveMode
from curobo._src.solver.solve_state import SolveState
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose


@pytest.fixture
def cpu_device_cfg():
    """Create a CPU device configuration for testing."""
    return DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)


@pytest.fixture
def cuda_device_cfg():
    """Create a CUDA device configuration for testing."""
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


@pytest.fixture
def goal_manager(cpu_device_cfg):
    """Create a GoalManager instance."""
    return GoalManager(cpu_device_cfg)


@pytest.fixture
def sample_solve_state():
    """Create a sample SolveState."""
    return SolveState(
        solve_type=SolveMode.BATCH,
        batch_size=4,
        num_envs=1,
        num_ik_seeds=32,
    )


@pytest.fixture
def sample_goal_poses(cpu_device_cfg):
    """Create sample goal poses."""
    batch_size = 4
    num_links = 1
    num_goalset = 1

    position = torch.randn(batch_size, 1, num_links, num_goalset, 3, **cpu_device_cfg.as_torch_dict())
    quaternion = torch.zeros(
        batch_size, 1, num_links, num_goalset, 4, **cpu_device_cfg.as_torch_dict()
    )
    quaternion[..., 0] = 1.0  # Unit quaternion

    return GoalToolPose(
        tool_frames=["ee_link"],
        position=position,
        quaternion=quaternion,
    )


class TestGoalManagerInitialization:
    """Test GoalManager initialization."""

    def test_init_basic(self, cpu_device_cfg):
        """Test basic initialization."""
        manager = GoalManager(cpu_device_cfg)
        assert manager.device_cfg == cpu_device_cfg
        assert manager._solve_state is None
        assert manager._goal_buffer is None

    def test_init_cuda(self, cuda_device_cfg):
        """Test initialization with CUDA."""
        manager = GoalManager(cuda_device_cfg)
        assert manager.device_cfg == cuda_device_cfg


class TestGoalManagerCreateGoalBuffer:
    """Test GoalManager.create_goal_buffer method."""

    def test_create_goal_buffer_minimal(self, goal_manager, sample_solve_state):
        """Test creating goal buffer with minimal args."""
        buffer = goal_manager.create_goal_buffer(sample_solve_state)

        assert isinstance(buffer, GoalRegistry)

    def test_create_goal_buffer_with_goal_poses(
        self, goal_manager, sample_solve_state, sample_goal_poses
    ):
        """Test creating goal buffer with goal poses."""
        buffer = goal_manager.create_goal_buffer(
            sample_solve_state,
            goal_tool_poses=sample_goal_poses,
        )

        assert buffer.link_goal_poses is not None

    def test_create_goal_buffer_with_current_js(
        self, goal_manager, sample_solve_state, cpu_device_cfg
    ):
        """Test creating goal buffer with current joint state."""
        batch_size = 4
        dof = 7
        current_js = JointState.from_position(
            torch.randn(batch_size, dof, **cpu_device_cfg.as_torch_dict())
        )

        buffer = goal_manager.create_goal_buffer(
            sample_solve_state,
            current_js=current_js,
        )

        assert buffer.current_js is not None

    def test_create_goal_buffer_with_goal_js(
        self, goal_manager, sample_solve_state, cpu_device_cfg
    ):
        """Test creating goal buffer with goal joint state."""
        batch_size = 4
        dof = 7
        goal_js = JointState.from_position(
            torch.randn(batch_size, dof, **cpu_device_cfg.as_torch_dict())
        )

        buffer = goal_manager.create_goal_buffer(
            sample_solve_state,
            goal_js=goal_js,
        )

        assert buffer.goal_js is not None


class TestGoalManagerUpdateGoalBuffer:
    """Test GoalManager.update_goal_buffer method."""

    def test_update_goal_buffer_first_call_creates_buffer(
        self, goal_manager, sample_solve_state, sample_goal_poses
    ):
        """Test that first update call creates buffer."""
        buffer, update_reference = goal_manager.update_goal_buffer(
            sample_solve_state,
            goal_tool_poses=sample_goal_poses,
        )

        assert isinstance(buffer, GoalRegistry)
        assert update_reference is True

    def test_update_goal_buffer_second_call_reuses(
        self, goal_manager, sample_solve_state, sample_goal_poses
    ):
        """Test that second update call reuses buffer when possible."""
        # First call
        buffer1, _ = goal_manager.update_goal_buffer(
            sample_solve_state,
            goal_tool_poses=sample_goal_poses,
        )

        # Second call with same solve_state
        buffer2, update_reference = goal_manager.update_goal_buffer(
            sample_solve_state,
            goal_tool_poses=sample_goal_poses,
        )

        # Should reuse buffer
        assert buffer2 is buffer1 or not update_reference


class TestGoalManagerBatchHelper:
    """Test GoalManager batch helper property."""

    def test_batch_helper_after_update(
        self, goal_manager, sample_solve_state, sample_goal_poses
    ):
        """Test batch_helper is set after update."""
        goal_manager.update_goal_buffer(
            sample_solve_state,
            goal_tool_poses=sample_goal_poses,
        )

        # batch_helper should be a tensor for indexing
        assert hasattr(goal_manager, 'batch_helper')


class TestGoalManagerSingleBatch:
    """Test GoalManager with single batch (batch_size=1)."""

    def test_single_batch_create_buffer(self, goal_manager, cpu_device_cfg):
        """Test creating goal buffer for single batch."""
        solve_state = SolveState(
            solve_type=SolveMode.SINGLE,
            batch_size=1,
            num_envs=1,
            num_ik_seeds=32,
        )

        position = torch.randn(1, 1, 1, 1, 3, **cpu_device_cfg.as_torch_dict())
        quaternion = torch.zeros(1, 1, 1, 1, 4, **cpu_device_cfg.as_torch_dict())
        quaternion[..., 0] = 1.0

        goal_poses = GoalToolPose(
            tool_frames=["ee_link"],
            position=position,
            quaternion=quaternion,
        )

        buffer = goal_manager.create_goal_buffer(solve_state, goal_tool_poses=goal_poses)

        assert isinstance(buffer, GoalRegistry)


class TestGoalManagerGoalsetMode:
    """Test GoalManager with goalset mode (multiple goals per problem)."""

    def test_goalset_create_buffer(self, goal_manager, cpu_device_cfg):
        """Test creating goal buffer for goalset mode."""
        num_goalset = 4
        solve_state = SolveState(
            solve_type=SolveMode.SINGLE,
            batch_size=1,
            num_envs=1,
            num_goalset=num_goalset,
            num_ik_seeds=32,
        )

        position = torch.randn(1, 1, 1, num_goalset, 3, **cpu_device_cfg.as_torch_dict())
        quaternion = torch.zeros(1, 1, 1, num_goalset, 4, **cpu_device_cfg.as_torch_dict())
        quaternion[..., 0] = 1.0

        goal_poses = GoalToolPose(
            tool_frames=["ee_link"],
            position=position,
            quaternion=quaternion,
        )

        buffer = goal_manager.create_goal_buffer(solve_state, goal_tool_poses=goal_poses)

        assert isinstance(buffer, GoalRegistry)


class TestGoalManagerCUDA:
    """Test GoalManager with CUDA tensors."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_buffer_cuda(self, cuda_device_cfg):
        """Test creating goal buffer on CUDA."""
        manager = GoalManager(cuda_device_cfg)
        solve_state = SolveState(
            solve_type=SolveMode.SINGLE,
            batch_size=1,
            num_envs=1,
            num_ik_seeds=32,
        )

        position = torch.randn(1, 1, 1, 1, 3, **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros(1, 1, 1, 1, 4, **cuda_device_cfg.as_torch_dict())
        quaternion[..., 0] = 1.0

        goal_poses = GoalToolPose(
            tool_frames=["ee_link"],
            position=position,
            quaternion=quaternion,
        )

        buffer = manager.create_goal_buffer(solve_state, goal_tool_poses=goal_poses)

        assert isinstance(buffer, GoalRegistry)


class TestGoalManagerMultipleLinks:
    """Test GoalManager with multiple tool frames."""

    def test_multiple_links_create_buffer(self, goal_manager, cpu_device_cfg):
        """Test creating goal buffer for multiple links."""
        batch_size = 2
        num_links = 3
        num_goalset = 1

        solve_state = SolveState(
            solve_type=SolveMode.BATCH,
            batch_size=batch_size,
            num_envs=1,
            num_ik_seeds=32,
        )

        position = torch.randn(
            batch_size, 1, num_links, num_goalset, 3, **cpu_device_cfg.as_torch_dict()
        )
        quaternion = torch.zeros(
            batch_size, 1, num_links, num_goalset, 4, **cpu_device_cfg.as_torch_dict()
        )
        quaternion[..., 0] = 1.0

        goal_poses = GoalToolPose(
            tool_frames=["link1", "link2", "link3"],
            position=position,
            quaternion=quaternion,
        )

        buffer = goal_manager.create_goal_buffer(solve_state, goal_tool_poses=goal_poses)

        assert isinstance(buffer, GoalRegistry)
        assert buffer.link_goal_poses is not None


class TestGoalManagerNumSeedsNone:
    """Test GoalManager behavior when num_seeds is None (line 82)."""

    def test_create_goal_buffer_num_seeds_none_raises(self, goal_manager, cpu_device_cfg):
        """Test that creating buffer without num_seeds raises error (line 82)."""
        solve_state = SolveState(
            solve_type=SolveMode.SINGLE,
            batch_size=1,
            num_envs=1,
            num_ik_seeds=None,  # No seeds set
        )

        with pytest.raises(ValueError, match="Number of seeds is not set"):
            goal_manager.create_goal_buffer(solve_state)


class TestGoalManagerUpdateGoal:
    """Test GoalManager.update_from_goal_registry method (lines 199-236)."""

    def test_update_goal_first_call(self, goal_manager, cpu_device_cfg):
        """Test update_goal creates buffer on first call (lines 220-228)."""
        solve_state = SolveState(
            solve_type=SolveMode.BATCH,
            batch_size=4,
            num_envs=1,
            num_ik_seeds=32,
        )

        # Create a simple goal registry
        goal = GoalRegistry.create_idx(
            pose_batch_size=4,
            multi_env=False,
            num_seeds=32,
            device_cfg=cpu_device_cfg,
        )

        buffer, update_reference = goal_manager.update_from_goal_registry(solve_state, goal)

        assert isinstance(buffer, GoalRegistry)
        assert update_reference is True

    def test_update_goal_reuses_buffer(self, goal_manager, cpu_device_cfg):
        """Test update_goal reuses buffer on subsequent calls (line 234)."""
        solve_state = SolveState(
            solve_type=SolveMode.BATCH,
            batch_size=4,
            num_envs=1,
            num_ik_seeds=32,
        )

        goal = GoalRegistry.create_idx(
            pose_batch_size=4,
            multi_env=False,
            num_seeds=32,
            device_cfg=cpu_device_cfg,
        )

        # First call
        buffer1, _ = goal_manager.update_from_goal_registry(solve_state, goal)

        # Second call with same solve_state
        buffer2, update_reference = goal_manager.update_from_goal_registry(solve_state, goal)

        assert update_reference is False

    def test_update_goal_with_goal_js_mismatch(self, goal_manager, cpu_device_cfg):
        """Test update_goal recreates buffer when goal_js changes (lines 204-205)."""
        solve_state = SolveState(
            solve_type=SolveMode.BATCH,
            batch_size=4,
            num_envs=1,
            num_ik_seeds=32,
        )

        # First call without goal_js
        goal1 = GoalRegistry.create_idx(
            pose_batch_size=4,
            multi_env=False,
            num_seeds=32,
            device_cfg=cpu_device_cfg,
        )
        buffer1, _ = goal_manager.update_from_goal_registry(solve_state, goal1)

        # Second call with goal_js
        goal2 = GoalRegistry.create_idx(
            pose_batch_size=4,
            multi_env=False,
            num_seeds=32,
            device_cfg=cpu_device_cfg,
        )
        goal2.goal_js = JointState.from_position(
            torch.randn(4, 7, **cpu_device_cfg.as_torch_dict())
        )

        buffer2, update_reference = goal_manager.update_from_goal_registry(solve_state, goal2)

        assert update_reference is True

    def test_update_goal_with_seed_goal_js_mismatch(self, goal_manager, cpu_device_cfg):
        """Test update_goal recreates buffer when seed_goal_js changes (lines 206-207)."""
        solve_state = SolveState(
            solve_type=SolveMode.BATCH,
            batch_size=4,
            num_envs=1,
            num_ik_seeds=32,
        )
        num_seeds = 32
        dof = 7

        # First call with seed_goal_js (needs shape [batch, num_seeds, dof])
        goal1 = GoalRegistry.create_idx(
            pose_batch_size=4,
            multi_env=False,
            num_seeds=num_seeds,
            device_cfg=cpu_device_cfg,
            seed_goal_state=JointState.from_position(
                torch.randn(4, num_seeds, dof, **cpu_device_cfg.as_torch_dict())
            ),
        )
        buffer1, _ = goal_manager.update_from_goal_registry(solve_state, goal1)

        # Second call without seed_goal_js
        goal2 = GoalRegistry.create_idx(
            pose_batch_size=4,
            multi_env=False,
            num_seeds=num_seeds,
            device_cfg=cpu_device_cfg,
        )

        buffer2, update_reference = goal_manager.update_from_goal_registry(solve_state, goal2)

        assert update_reference is True

    def test_update_goal_with_idxs_seed_goal_js(self, goal_manager, cpu_device_cfg):
        """Test update_goal preserves idxs_seed_goal_js (lines 229-230)."""
        solve_state = SolveState(
            solve_type=SolveMode.BATCH,
            batch_size=4,
            num_envs=1,
            num_ik_seeds=32,
        )

        goal = GoalRegistry.create_idx(
            pose_batch_size=4,
            multi_env=False,
            num_seeds=32,
            device_cfg=cpu_device_cfg,
        )
        goal.idxs_seed_goal_js = torch.arange(4, device=cpu_device_cfg.device)

        buffer, _ = goal_manager.update_from_goal_registry(solve_state, goal)

        assert buffer.idxs_seed_goal_js is not None

    def test_update_goal_with_seed_enable_implicit(self, goal_manager, cpu_device_cfg):
        """Test update_goal preserves seed_enable_implicit_goal_js (lines 231-232)."""
        solve_state = SolveState(
            solve_type=SolveMode.BATCH,
            batch_size=4,
            num_envs=1,
            num_ik_seeds=32,
        )

        goal = GoalRegistry.create_idx(
            pose_batch_size=4,
            multi_env=False,
            num_seeds=32,
            device_cfg=cpu_device_cfg,
        )
        goal.seed_enable_implicit_goal_js = torch.ones(4, device=cpu_device_cfg.device)

        buffer, _ = goal_manager.update_from_goal_registry(solve_state, goal)

        assert buffer.seed_enable_implicit_goal_js is not None


class TestGoalManagerPropertyErrors:
    """Test GoalManager property error cases (lines 255, 257, 283, 285, 300, 302, 316-318, 333-335, 341)."""

    def test_goal_buffer_not_initialized_raises(self, goal_manager):
        """Test goal_buffer property raises when not initialized (lines 333-335)."""
        with pytest.raises(ValueError, match="Goal buffer has not been initialized"):
            _ = goal_manager.goal_buffer

    def test_solve_state_not_initialized_raises(self, goal_manager):
        """Test solve_state property raises when not initialized (lines 341)."""
        with pytest.raises(ValueError, match="Solve state has not been initialized"):
            _ = goal_manager.solve_state

    def test_update_goal_tool_poses_buffer_not_init(self, goal_manager, sample_goal_poses):
        """Test update_goal_tool_poses raises when buffer not init (line 255)."""
        with pytest.raises(ValueError, match="Goal buffer has not been initialized"):
            goal_manager.update_goal_tool_poses(sample_goal_poses)

    def test_update_current_state_buffer_not_init(self, goal_manager, cpu_device_cfg):
        """Test update_current_state raises when buffer not init (line 283)."""
        current_state = JointState.from_position(
            torch.randn(4, 7, **cpu_device_cfg.as_torch_dict())
        )
        with pytest.raises(ValueError, match="Goal buffer has not been initialized"):
            goal_manager.update_current_state(current_state)

    def test_update_goal_state_buffer_not_init(self, goal_manager, cpu_device_cfg):
        """Test update_goal_state raises when buffer not init (line 300)."""
        goal_state = JointState.from_position(
            torch.randn(4, 7, **cpu_device_cfg.as_torch_dict())
        )
        with pytest.raises(ValueError, match="Goal buffer has not been initialized"):
            goal_manager.update_goal_state(goal_state)


class TestGoalManagerBatchSizeGetters:
    """Test batch size getter methods (lines 351-365)."""

    def test_get_batch_size_not_initialized(self, goal_manager):
        """Test get_batch_size returns 0 when not initialized (lines 351-352)."""
        assert goal_manager.get_batch_size() == 0

    def test_get_ik_batch_size_not_initialized(self, goal_manager):
        """Test get_ik_batch_size returns 0 when not initialized (lines 357-358)."""
        assert goal_manager.get_ik_batch_size() == 0

    def test_get_trajopt_batch_size_not_initialized(self, goal_manager):
        """Test get_trajopt_batch_size returns 0 when not initialized (lines 363-364)."""
        assert goal_manager.get_trajopt_batch_size() == 0

    def test_get_batch_size_after_init(self, goal_manager, sample_solve_state, sample_goal_poses):
        """Test get_batch_size after initialization (line 353)."""
        goal_manager.update_goal_buffer(sample_solve_state, goal_tool_poses=sample_goal_poses)
        assert goal_manager.get_batch_size() == 4

    def test_get_ik_batch_size_after_init(self, goal_manager, sample_solve_state, sample_goal_poses):
        """Test get_ik_batch_size after initialization (line 359)."""
        goal_manager.update_goal_buffer(sample_solve_state, goal_tool_poses=sample_goal_poses)
        result = goal_manager.get_ik_batch_size()
        assert result >= 0

    def test_get_trajopt_batch_size_after_init(self, goal_manager, sample_solve_state, sample_goal_poses):
        """Test get_trajopt_batch_size after initialization (line 365).

        Note: Returns 0 when num_trajopt_seeds is None (now handles None gracefully).
        """
        goal_manager.update_goal_buffer(sample_solve_state, goal_tool_poses=sample_goal_poses)
        result = goal_manager.get_trajopt_batch_size()
        # Returns 0 when num_trajopt_seeds is not set
        assert result == 0

    def test_get_trajopt_batch_size_with_seeds(self, goal_manager, sample_goal_poses, cpu_device_cfg):
        """Test get_trajopt_batch_size with num_trajopt_seeds set."""
        solve_state = SolveState(
            solve_type=SolveMode.BATCH,
            batch_size=4,
            num_envs=1,
            num_ik_seeds=32,
            num_trajopt_seeds=4,
        )
        goal_manager.update_goal_buffer(solve_state, goal_tool_poses=sample_goal_poses)
        result = goal_manager.get_trajopt_batch_size()
        assert result == 4 * 4  # num_trajopt_seeds * batch_size


class TestGoalManagerPaddedGoalset:
    """Test _get_padded_goalset_for_links (lines 389, 406-438, 455-489)."""

    def test_padded_goalset_none_inputs(self, cpu_device_cfg):
        """Test padding returns None when inputs are None (line 389)."""
        result = GoalManager._get_padded_goalset_for_links(
            solve_state=SolveState(
                solve_type=SolveMode.SINGLE,
                batch_size=1,
                num_envs=1,
                num_ik_seeds=32,
            ),
            current_solve_state=SolveState(
                solve_type=SolveMode.SINGLE,
                batch_size=1,
                num_envs=1,
                num_goalset=4,
                num_ik_seeds=32,
            ),
            current_goal_buffer=GoalRegistry.create_idx(
                pose_batch_size=1,
                multi_env=False,
                num_seeds=32,
                device_cfg=cpu_device_cfg,
            ),
            links_goal_pose=None,
        )
        assert result is None

    def test_padded_goalset_to_single(self, goal_manager, cpu_device_cfg):
        """Test padding from GOALSET to SINGLE type (lines 406-438)."""
        num_goalset = 4

        # First create goalset buffer
        goalset_solve_state = SolveState(
            solve_type=SolveMode.SINGLE,
            batch_size=1,
            num_envs=1,
            num_goalset=num_goalset,
            num_ik_seeds=32,
        )

        position = torch.randn(1, 1, 1, num_goalset, 3, **cpu_device_cfg.as_torch_dict())
        quaternion = torch.zeros(1, 1, 1, num_goalset, 4, **cpu_device_cfg.as_torch_dict())
        quaternion[..., 0] = 1.0

        goalset_poses = GoalToolPose(
            tool_frames=["ee_link"],
            position=position,
            quaternion=quaternion,
        )

        goal_manager.update_goal_buffer(goalset_solve_state, goal_tool_poses=goalset_poses)

        # Now update with single solve state
        single_solve_state = SolveState(
            solve_type=SolveMode.SINGLE,
            batch_size=1,
            num_envs=1,
            num_ik_seeds=32,
        )

        single_position = torch.randn(1, 1, 1, 1, 3, **cpu_device_cfg.as_torch_dict())
        single_quaternion = torch.zeros(1, 1, 1, 1, 4, **cpu_device_cfg.as_torch_dict())
        single_quaternion[..., 0] = 1.0

        single_poses = GoalToolPose(
            tool_frames=["ee_link"],
            position=single_position,
            quaternion=single_quaternion,
        )

        buffer, update_reference = goal_manager.update_goal_buffer(
            single_solve_state, goal_tool_poses=single_poses
        )

        # Should reuse buffer with padding
        assert buffer is not None

    def test_padded_goalset_mismatched_links_returns_none(self, cpu_device_cfg):
        """Test padding returns None when tool_frames don't match (lines 406-409)."""
        # Create current goal buffer with ee_link
        current_goal_buffer = GoalRegistry.create_idx(
            pose_batch_size=1,
            multi_env=False,
            num_seeds=32,
            device_cfg=cpu_device_cfg,
        )

        position = torch.randn(1, 1, 1, 4, 3, **cpu_device_cfg.as_torch_dict())
        quaternion = torch.zeros(1, 1, 1, 4, 4, **cpu_device_cfg.as_torch_dict())
        quaternion[..., 0] = 1.0

        current_goal_buffer.link_goal_poses = GoalToolPose(
            tool_frames=["ee_link"],
            position=position,
            quaternion=quaternion,
        )

        # Create new goal poses with different tool_frames
        new_position = torch.randn(1, 1, 1, 1, 3, **cpu_device_cfg.as_torch_dict())
        new_quaternion = torch.zeros(1, 1, 1, 1, 4, **cpu_device_cfg.as_torch_dict())
        new_quaternion[..., 0] = 1.0

        new_goal_poses = GoalToolPose(
            tool_frames=["other_link"],  # Different link
            position=new_position,
            quaternion=new_quaternion,
        )

        result = GoalManager._get_padded_goalset_for_links(
            solve_state=SolveState(
                solve_type=SolveMode.SINGLE,
                batch_size=1,
                num_envs=1,
                num_ik_seeds=32,
            ),
            current_solve_state=SolveState(
                solve_type=SolveMode.SINGLE,
                batch_size=1,
                num_envs=1,
                num_goalset=4,
                num_ik_seeds=32,
            ),
            current_goal_buffer=current_goal_buffer,
            links_goal_pose=new_goal_poses,
        )
        assert result is None

    def test_padded_goalset_to_goalset(self, goal_manager, cpu_device_cfg):
        """Test padding from larger GOALSET to smaller GOALSET (lines 455-489)."""
        large_n_goalset = 8
        small_n_goalset = 4

        # First create large goalset buffer
        large_solve_state = SolveState(
            solve_type=SolveMode.SINGLE,
            batch_size=1,
            num_envs=1,
            num_goalset=large_n_goalset,
            num_ik_seeds=32,
        )

        position = torch.randn(1, 1, 1, large_n_goalset, 3, **cpu_device_cfg.as_torch_dict())
        quaternion = torch.zeros(1, 1, 1, large_n_goalset, 4, **cpu_device_cfg.as_torch_dict())
        quaternion[..., 0] = 1.0

        large_poses = GoalToolPose(
            tool_frames=["ee_link"],
            position=position,
            quaternion=quaternion,
        )

        goal_manager.update_goal_buffer(large_solve_state, goal_tool_poses=large_poses)

        # Now update with smaller goalset
        small_solve_state = SolveState(
            solve_type=SolveMode.SINGLE,
            batch_size=1,
            num_envs=1,
            num_goalset=small_n_goalset,
            num_ik_seeds=32,
        )

        small_position = torch.randn(1, 1, 1, small_n_goalset, 3, **cpu_device_cfg.as_torch_dict())
        small_quaternion = torch.zeros(
            1, 1, 1, small_n_goalset, 4, **cpu_device_cfg.as_torch_dict()
        )
        small_quaternion[..., 0] = 1.0

        small_poses = GoalToolPose(
            tool_frames=["ee_link"],
            position=small_position,
            quaternion=small_quaternion,
        )

        buffer, update_reference = goal_manager.update_goal_buffer(
            small_solve_state, goal_tool_poses=small_poses
        )

        # Should reuse buffer with padding
        assert buffer is not None

    def test_padded_goalset_3d_pose(self, cpu_device_cfg):
        """Test padding with 3D pose shapes (lines 418-419, 425-426)."""
        current_goal_buffer = GoalRegistry.create_idx(
            pose_batch_size=1,
            multi_env=False,
            num_seeds=32,
            device_cfg=cpu_device_cfg,
        )

        position = torch.randn(1, 1, 1, 1, 3, **cpu_device_cfg.as_torch_dict())
        quaternion = torch.zeros(1, 1, 1, 1, 4, **cpu_device_cfg.as_torch_dict())
        quaternion[..., 0] = 1.0

        current_goal_buffer.link_goal_poses = GoalToolPose(
            tool_frames=["ee_link"],
            position=position,
            quaternion=quaternion,
        )

        new_position = torch.randn(1, 1, 1, 1, 3, **cpu_device_cfg.as_torch_dict())
        new_quaternion = torch.zeros(1, 1, 1, 1, 4, **cpu_device_cfg.as_torch_dict())
        new_quaternion[..., 0] = 1.0

        new_goal_poses = GoalToolPose(
            tool_frames=["ee_link"],
            position=new_position,
            quaternion=new_quaternion,
        )

        result = GoalManager._get_padded_goalset_for_links(
            solve_state=SolveState(
                solve_type=SolveMode.SINGLE,
                batch_size=1,
                num_envs=1,
                num_goalset=1,
                num_ik_seeds=32,
            ),
            current_solve_state=SolveState(
                solve_type=SolveMode.SINGLE,
                batch_size=1,
                num_envs=1,
                num_goalset=1,
                num_ik_seeds=32,
            ),
            current_goal_buffer=current_goal_buffer,
            links_goal_pose=new_goal_poses,
        )

        # Result should be padded GoalToolPose
        assert result is not None

    def test_padded_goalset_batch_goalset(self, goal_manager, cpu_device_cfg):
        """Test padding from BATCH_GOALSET to BATCH (lines 398-400)."""
        batch_size = 2
        num_goalset = 4

        # First create batch goalset buffer
        batch_goalset_state = SolveState(
            solve_type=SolveMode.BATCH,
            batch_size=batch_size,
            num_envs=1,
            num_goalset=num_goalset,
            num_ik_seeds=32,
        )

        position = torch.randn(
            batch_size, 1, 1, num_goalset, 3, **cpu_device_cfg.as_torch_dict()
        )
        quaternion = torch.zeros(
            batch_size, 1, 1, num_goalset, 4, **cpu_device_cfg.as_torch_dict()
        )
        quaternion[..., 0] = 1.0

        goalset_poses = GoalToolPose(
            tool_frames=["ee_link"],
            position=position,
            quaternion=quaternion,
        )

        goal_manager.update_goal_buffer(batch_goalset_state, goal_tool_poses=goalset_poses)

        # Now update with BATCH solve state
        batch_solve_state = SolveState(
            solve_type=SolveMode.BATCH,
            batch_size=batch_size,
            num_envs=1,
            num_ik_seeds=32,
        )

        batch_position = torch.randn(batch_size, 1, 1, 1, 3, **cpu_device_cfg.as_torch_dict())
        batch_quaternion = torch.zeros(
            batch_size, 1, 1, 1, 4, **cpu_device_cfg.as_torch_dict()
        )
        batch_quaternion[..., 0] = 1.0

        batch_poses = GoalToolPose(
            tool_frames=["ee_link"],
            position=batch_position,
            quaternion=batch_quaternion,
        )

        buffer, update_reference = goal_manager.update_goal_buffer(
            batch_solve_state, goal_tool_poses=batch_poses
        )

        assert buffer is not None


class TestGoalManagerUpdateGoalBufferNewFields:
    """Test update_goal_buffer when new fields are added (lines 137-140)."""

    def test_update_goal_buffer_adds_goal_js(self, goal_manager, sample_solve_state, sample_goal_poses, cpu_device_cfg):
        """Test update recreates buffer when goal_js is added (line 137)."""
        # First call without goal_js
        buffer1, _ = goal_manager.update_goal_buffer(
            sample_solve_state,
            goal_tool_poses=sample_goal_poses,
        )
        assert buffer1.goal_js is None

        # Second call with goal_js
        goal_js = JointState.from_position(
            torch.randn(4, 7, **cpu_device_cfg.as_torch_dict())
        )
        buffer2, update_reference = goal_manager.update_goal_buffer(
            sample_solve_state,
            goal_tool_poses=sample_goal_poses,
            goal_js=goal_js,
        )

        assert update_reference is True

    def test_update_goal_buffer_adds_current_js(self, goal_manager, sample_solve_state, sample_goal_poses, cpu_device_cfg):
        """Test update recreates buffer when current_js is added (line 139)."""
        # First call without current_js
        buffer1, _ = goal_manager.update_goal_buffer(
            sample_solve_state,
            goal_tool_poses=sample_goal_poses,
        )

        # Second call with current_js
        current_js = JointState.from_position(
            torch.randn(4, 7, **cpu_device_cfg.as_torch_dict())
        )
        buffer2, update_reference = goal_manager.update_goal_buffer(
            sample_solve_state,
            goal_tool_poses=sample_goal_poses,
            current_js=current_js,
        )

        assert update_reference is True

    def test_update_goal_buffer_adds_seed_goal_js(self, goal_manager, sample_solve_state, sample_goal_poses, cpu_device_cfg):
        """Test update recreates buffer when seed_goal_js is added (line 140)."""
        num_seeds = 32
        dof = 7
        # First call without seed_goal_js
        buffer1, _ = goal_manager.update_goal_buffer(
            sample_solve_state,
            goal_tool_poses=sample_goal_poses,
        )

        # Second call with seed_goal_js (needs shape [batch, num_seeds, dof])
        seed_goal_js = JointState.from_position(
            torch.randn(4, num_seeds, dof, **cpu_device_cfg.as_torch_dict())
        )
        buffer2, update_reference = goal_manager.update_goal_buffer(
            sample_solve_state,
            goal_tool_poses=sample_goal_poses,
            seed_goal_js=seed_goal_js,
        )

        assert update_reference is True

    def test_update_goal_buffer_use_implicit_goal(self, goal_manager, sample_solve_state, sample_goal_poses, cpu_device_cfg):
        """Test update_goal_buffer with use_implicit_goal (lines 181-182)."""
        num_seeds = 32
        dof = 7
        # seed_goal_js needs shape [batch, num_seeds, dof]
        seed_goal_js = JointState.from_position(
            torch.randn(4, num_seeds, dof, **cpu_device_cfg.as_torch_dict())
        )

        buffer, _ = goal_manager.update_goal_buffer(
            sample_solve_state,
            goal_tool_poses=sample_goal_poses,
            seed_goal_js=seed_goal_js,
            use_implicit_goal=True,
        )

        assert buffer is not None

