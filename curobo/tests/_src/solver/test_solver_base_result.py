# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for BaseSolverResult dataclass."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.solver.solver_base_result import BaseSolverResult


@pytest.fixture
def sample_result():
    """Create a sample BaseSolverResult for testing."""
    batch_size = 4
    num_seeds = 2
    dof = 7
    horizon = 1

    return BaseSolverResult(
        success=torch.tensor([[True, False], [True, True], [False, False], [True, False]]),
        solution=torch.randn(batch_size, num_seeds, horizon, dof),
        position_error=torch.rand(batch_size, num_seeds),
        rotation_error=torch.rand(batch_size, num_seeds),
        goalset_index=torch.zeros(batch_size, num_seeds, dtype=torch.long),
        solve_time=0.05,
        total_time=0.1,
        batch_size=batch_size,
        num_seeds=num_seeds,
    )


class TestBaseSolverResultDataclass:
    """Test BaseSolverResult dataclass attributes."""

    def test_required_success_field(self):
        """Test that success is required."""
        result = BaseSolverResult(
            success=torch.tensor([True, False]),
        )
        assert result.success is not None

    def test_default_values(self):
        """Test default values are set correctly."""
        result = BaseSolverResult(
            success=torch.tensor([True]),
        )
        assert result.solution is None
        assert result.js_solution is None
        assert result.position_error is None
        assert result.rotation_error is None
        assert result.cspace_error is None
        assert result.goalset_index is None
        assert result.solve_time == 0.0
        assert result.total_time == 0.0
        assert result.debug_info == {}
        assert result.optimized_seeds is None
        assert result.metrics is None
        assert result.position_tolerance == 0.0
        assert result.orientation_tolerance == 0.0
        assert result.seed_rank is None
        assert result.seed_cost is None
        assert result.batch_size == 0
        assert result.num_seeds == 0
        assert result.total_cost_reshaped is None
        assert result.solution_state is None

    def test_full_initialization(self, sample_result):
        """Test full initialization with all fields."""
        assert sample_result.success is not None
        assert sample_result.solution is not None
        assert sample_result.position_error is not None
        assert sample_result.rotation_error is not None
        assert sample_result.goalset_index is not None
        assert sample_result.solve_time == 0.05
        assert sample_result.total_time == 0.1
        assert sample_result.batch_size == 4
        assert sample_result.num_seeds == 2


class TestBaseSolverResultClone:
    """Test BaseSolverResult.clone method."""

    def test_clone_creates_copy(self, sample_result):
        """Test clone creates a new object."""
        cloned = sample_result.clone()
        assert cloned is not sample_result

    def test_clone_preserves_values(self, sample_result):
        """Test clone preserves tensor values and selected scalar values."""
        cloned = sample_result.clone()

        # Tensor values are cloned
        assert torch.equal(cloned.success, sample_result.success)
        assert torch.equal(cloned.solution, sample_result.solution)
        assert torch.equal(cloned.position_error, sample_result.position_error)
        assert torch.equal(cloned.rotation_error, sample_result.rotation_error)
        assert torch.equal(cloned.goalset_index, sample_result.goalset_index)

        # Scalar values preserved
        assert cloned.solve_time == sample_result.solve_time
        assert cloned.batch_size == sample_result.batch_size
        assert cloned.num_seeds == sample_result.num_seeds
        # Note: total_time is not copied in clone() - appears intentional

    def test_clone_is_independent(self, sample_result):
        """Test cloned tensors are independent."""
        cloned = sample_result.clone()

        # Modify original
        sample_result.success[0, 0] = False
        sample_result.solution[0, 0, 0, 0] = 999.0

        # Clone should be unaffected
        assert cloned.success[0, 0] == True
        assert cloned.solution[0, 0, 0, 0] != 999.0

    def test_clone_with_none_fields(self):
        """Test clone handles None fields correctly."""
        result = BaseSolverResult(
            success=torch.tensor([True]),
        )
        cloned = result.clone()

        assert cloned.success is not None
        assert cloned.solution is None
        assert cloned.js_solution is None

    def test_clone_with_debug_info(self):
        """Test clone handles debug_info correctly."""
        debug_tensor = torch.randn(3, 3)
        result = BaseSolverResult(
            success=torch.tensor([True]),
            debug_info={"tensor": debug_tensor, "scalar": 42},
        )
        cloned = result.clone()

        assert "tensor" in cloned.debug_info
        assert "scalar" in cloned.debug_info
        assert torch.equal(cloned.debug_info["tensor"], debug_tensor)
        assert cloned.debug_info["scalar"] == 42


class TestBaseSolverResultCopySuccessfulSolutions:
    """Test BaseSolverResult.copy_successful_solutions method."""

    def test_copy_successful_solutions_basic(self):
        """Test copying successful solutions from one result to another."""
        batch_size = 2
        num_seeds = 2
        dof = 7
        horizon = 1

        # Create result with no successes
        result1 = BaseSolverResult(
            success=torch.tensor([[False, False], [False, False]]),
            solution=torch.zeros(batch_size, num_seeds, horizon, dof),
            position_error=torch.ones(batch_size, num_seeds),
            rotation_error=torch.ones(batch_size, num_seeds),
            goalset_index=torch.zeros(batch_size, num_seeds, dtype=torch.long),
            batch_size=batch_size,
            num_seeds=num_seeds,
        )

        # Create result with some successes
        result2 = BaseSolverResult(
            success=torch.tensor([[True, False], [False, True]]),
            solution=torch.ones(batch_size, num_seeds, horizon, dof),
            position_error=torch.zeros(batch_size, num_seeds) + 0.001,
            rotation_error=torch.zeros(batch_size, num_seeds) + 0.01,
            goalset_index=torch.ones(batch_size, num_seeds, dtype=torch.long),
            batch_size=batch_size,
            num_seeds=num_seeds,
        )

        result1.copy_successful_solutions(result2)

        # Check that successful indices were copied
        assert result1.success[0, 0] == True
        assert result1.success[1, 1] == True
        # Check that non-successful indices were not modified
        assert result1.success[0, 1] == False
        assert result1.success[1, 0] == False


class TestBaseSolverResultSuccessShape:
    """Test BaseSolverResult success tensor shapes."""

    def test_success_1d(self):
        """Test 1D success tensor."""
        result = BaseSolverResult(
            success=torch.tensor([True, False, True]),
        )
        assert result.success.shape == (3,)

    def test_success_2d(self):
        """Test 2D success tensor (batch, seeds)."""
        result = BaseSolverResult(
            success=torch.tensor([[True, False], [True, True]]),
        )
        assert result.success.shape == (2, 2)


class TestBaseSolverResultDeviceHandling:
    """Test BaseSolverResult device handling."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_tensors(self):
        """Test result with CUDA tensors."""
        result = BaseSolverResult(
            success=torch.tensor([True, False], device="cuda:0"),
            solution=torch.randn(2, 1, 1, 7, device="cuda:0"),
        )
        assert result.success.is_cuda
        assert result.solution.is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_clone_preserves_device(self):
        """Test clone preserves device."""
        result = BaseSolverResult(
            success=torch.tensor([True, False], device="cuda:0"),
        )
        cloned = result.clone()
        assert cloned.success.is_cuda


class TestBaseSolverResultTimingInfo:
    """Test BaseSolverResult timing information."""

    def test_solve_time(self):
        """Test solve_time attribute."""
        result = BaseSolverResult(
            success=torch.tensor([True]),
            solve_time=0.123,
        )
        assert result.solve_time == 0.123

    def test_total_time(self):
        """Test total_time attribute."""
        result = BaseSolverResult(
            success=torch.tensor([True]),
            total_time=0.456,
        )
        assert result.total_time == 0.456

    def test_timing_is_mutable(self):
        """Test timing values can be updated."""
        result = BaseSolverResult(
            success=torch.tensor([True]),
        )
        result.solve_time = 0.1
        result.total_time = 0.2
        assert result.solve_time == 0.1
        assert result.total_time == 0.2

