# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for SolveState and related classes."""

# CuRobo
from curobo._src.solver.solve_mode import SolveMode
from curobo._src.solver.solve_state import MotionPlanSolveState, SolveState


class TestSolveMode:
    """Test SolveMode enum values used by SolveState."""

    def test_single_value(self):
        assert SolveMode.SINGLE.value == "single"

    def test_batch_value(self):
        assert SolveMode.BATCH.value == "batch"

    def test_multi_env_value(self):
        assert SolveMode.MULTI_ENV.value == "multi_env"

    def test_enum_members_count(self):
        assert len(SolveMode) == 3


class TestSolveStateInitialization:
    """Test SolveState initialization."""

    def test_minimal_init(self):
        state = SolveState(
            solve_type=SolveMode.SINGLE,
            batch_size=1,
            num_envs=1,
        )
        assert state.solve_type == SolveMode.SINGLE
        assert state.batch_size == 1
        assert state.num_envs == 1

    def test_default_values(self):
        state = SolveState(
            solve_type=SolveMode.SINGLE,
            batch_size=1,
            num_envs=1,
        )
        assert state.num_goalset == 1
        assert state.multi_env is False
        assert state.batch_mode is False
        assert state.num_graph_seeds is None
        assert state.num_trajopt_seeds is None
        assert state.tool_frames is None

    def test_full_init(self):
        state = SolveState(
            solve_type=SolveMode.BATCH,
            batch_size=10,
            num_envs=5,
            num_goalset=3,
            num_ik_seeds=32,
            num_graph_seeds=16,
            num_trajopt_seeds=8,
            tool_frames=["ee_link"],
        )
        assert state.solve_type == SolveMode.BATCH
        assert state.batch_size == 10
        assert state.num_envs == 5
        assert state.num_goalset == 3
        assert state.num_ik_seeds == 32
        assert state.num_graph_seeds == 16
        assert state.num_trajopt_seeds == 8
        assert state.tool_frames == ["ee_link"]


class TestSolveStatePostInit:
    """Test SolveState __post_init__ behavior."""

    def test_multi_env_false_when_num_envs_1(self):
        state = SolveState(
            solve_type=SolveMode.SINGLE,
            batch_size=1,
            num_envs=1,
        )
        assert state.multi_env is False

    def test_multi_env_true_when_num_envs_greater_than_1(self):
        state = SolveState(
            solve_type=SolveMode.MULTI_ENV,
            batch_size=4,
            num_envs=4,
        )
        assert state.multi_env is True

    def test_batch_mode_false_when_batch_size_1(self):
        state = SolveState(
            solve_type=SolveMode.SINGLE,
            batch_size=1,
            num_envs=1,
        )
        assert state.batch_mode is False

    def test_batch_mode_true_when_batch_size_greater_than_1(self):
        state = SolveState(
            solve_type=SolveMode.BATCH,
            batch_size=4,
            num_envs=1,
        )
        assert state.batch_mode is True

    def test_num_seeds_from_ik_seeds(self):
        state = SolveState(
            solve_type=SolveMode.SINGLE,
            batch_size=1,
            num_envs=1,
            num_ik_seeds=32,
        )
        assert state.num_seeds == 32

    def test_num_seeds_from_trajopt_seeds(self):
        state = SolveState(
            solve_type=SolveMode.SINGLE,
            batch_size=1,
            num_envs=1,
            num_trajopt_seeds=16,
        )
        assert state.num_seeds == 16

    def test_num_seeds_from_graph_seeds(self):
        state = SolveState(
            solve_type=SolveMode.SINGLE,
            batch_size=1,
            num_envs=1,
            num_graph_seeds=8,
        )
        assert state.num_seeds == 8

    def test_num_seeds_priority(self):
        state = SolveState(
            solve_type=SolveMode.SINGLE,
            batch_size=1,
            num_envs=1,
            num_ik_seeds=32,
            num_trajopt_seeds=16,
            num_graph_seeds=8,
        )
        assert state.num_seeds == 32


class TestSolveStateClone:
    """Test SolveState clone method."""

    def test_clone_creates_copy(self):
        original = SolveState(
            solve_type=SolveMode.BATCH,
            batch_size=10,
            num_envs=1,
            num_ik_seeds=32,
        )
        cloned = original.clone()
        assert cloned is not original

    def test_clone_preserves_values(self):
        original = SolveState(
            solve_type=SolveMode.MULTI_ENV,
            batch_size=10,
            num_envs=5,
            num_goalset=3,
            num_ik_seeds=32,
            num_graph_seeds=16,
            num_trajopt_seeds=8,
            tool_frames=["ee_link", "tool_link"],
        )
        cloned = original.clone()

        assert cloned.solve_type == original.solve_type
        assert cloned.batch_size == original.batch_size
        assert cloned.num_envs == original.num_envs
        assert cloned.num_goalset == original.num_goalset
        assert cloned.num_ik_seeds == original.num_ik_seeds
        assert cloned.num_graph_seeds == original.num_graph_seeds
        assert cloned.num_trajopt_seeds == original.num_trajopt_seeds
        assert cloned.tool_frames == original.tool_frames

    def test_clone_is_independent(self):
        original = SolveState(
            solve_type=SolveMode.SINGLE,
            batch_size=1,
            num_envs=1,
            num_ik_seeds=32,
            tool_frames=["ee_link"],
        )
        cloned = original.clone()
        cloned.batch_size = 100

        assert original.batch_size == 1
        assert cloned.batch_size == 100


class TestSolveStateBatchSizeMethods:
    """Test SolveState batch size methods."""

    def test_get_batch_size(self):
        state = SolveState(
            solve_type=SolveMode.BATCH,
            batch_size=10,
            num_envs=1,
            num_ik_seeds=32,
        )
        assert state.get_batch_size() == 32 * 10

    def test_get_ik_batch_size(self):
        state = SolveState(
            solve_type=SolveMode.BATCH,
            batch_size=10,
            num_envs=1,
            num_ik_seeds=32,
        )
        assert state.get_ik_batch_size() == 32 * 10

    def test_get_trajopt_batch_size(self):
        state = SolveState(
            solve_type=SolveMode.BATCH,
            batch_size=10,
            num_envs=1,
            num_trajopt_seeds=8,
        )
        assert state.get_trajopt_batch_size() == 8 * 10

    def test_get_batch_size_single(self):
        state = SolveState(
            solve_type=SolveMode.SINGLE,
            batch_size=1,
            num_envs=1,
            num_ik_seeds=32,
        )
        assert state.get_batch_size() == 32


class TestMotionPlanSolveState:
    """Test MotionPlanSolveState dataclass."""

    def test_initialization(self):
        ik_state = SolveState(
            solve_type=SolveMode.SINGLE,
            batch_size=1,
            num_envs=1,
            num_ik_seeds=32,
        )
        trajopt_state = SolveState(
            solve_type=SolveMode.SINGLE,
            batch_size=1,
            num_envs=1,
            num_trajopt_seeds=8,
        )

        motion_plan_state = MotionPlanSolveState(
            solve_type=SolveMode.SINGLE,
            ik_solve_state=ik_state,
            trajopt_solve_state=trajopt_state,
        )

        assert motion_plan_state.solve_type == SolveMode.SINGLE
        assert motion_plan_state.ik_solve_state == ik_state
        assert motion_plan_state.trajopt_solve_state == trajopt_state

    def test_motion_plan_with_different_solve_types(self):
        ik_state = SolveState(
            solve_type=SolveMode.BATCH,
            batch_size=10,
            num_envs=1,
            num_ik_seeds=32,
        )
        trajopt_state = SolveState(
            solve_type=SolveMode.BATCH,
            batch_size=10,
            num_envs=1,
            num_trajopt_seeds=4,
        )

        motion_plan_state = MotionPlanSolveState(
            solve_type=SolveMode.BATCH,
            ik_solve_state=ik_state,
            trajopt_solve_state=trajopt_state,
        )

        assert motion_plan_state.ik_solve_state.get_ik_batch_size() == 320
        assert motion_plan_state.trajopt_solve_state.get_trajopt_batch_size() == 40
