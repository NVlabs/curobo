# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for TrajOptSolverResult dataclass."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState


@pytest.fixture
def sample_trajopt_result():
    """Create a sample TrajOptSolverResult for testing."""
    batch_size = 2
    num_seeds = 2
    dof = 7
    horizon = 32

    return TrajOptSolverResult(
        success=torch.tensor([[True, False], [True, True]]),
        solution=torch.randn(batch_size, num_seeds, horizon, dof),
        position_error=torch.rand(batch_size, num_seeds),
        rotation_error=torch.rand(batch_size, num_seeds),
        goalset_index=torch.zeros(batch_size, num_seeds, dtype=torch.long),
        solve_time=0.1,
        total_time=0.2,
        batch_size=batch_size,
        num_seeds=num_seeds,
    )


class TestTrajOptSolverResultDataclass:
    """Test TrajOptSolverResult dataclass attributes."""

    def test_required_success_field(self):
        """Test that success is required."""
        result = TrajOptSolverResult(
            success=torch.tensor([True, False]),
        )
        assert result.success is not None

    def test_default_solution_is_none(self):
        """Test default solution is None."""
        result = TrajOptSolverResult(
            success=torch.tensor([True]),
        )
        assert result.solution is None

    def test_default_js_solution_is_none(self):
        """Test default js_solution is None."""
        result = TrajOptSolverResult(
            success=torch.tensor([True]),
        )
        assert result.js_solution is None

    def test_default_interpolated_trajectory_is_none(self):
        """Test default interpolated_trajectory is None."""
        result = TrajOptSolverResult(
            success=torch.tensor([True]),
        )
        assert result.interpolated_trajectory is None

    def test_default_interpolated_last_tstep_is_none(self):
        """Test default interpolated_last_tstep is None."""
        result = TrajOptSolverResult(
            success=torch.tensor([True]),
        )
        assert result.interpolated_last_tstep is None

    def test_default_interpolated_metrics_is_none(self):
        """Test default interpolated_metrics is None."""
        result = TrajOptSolverResult(
            success=torch.tensor([True]),
        )
        assert result.interpolated_metrics is None

    def test_has_trajectory_dt_bounds(self):
        """Test maximum/minimum trajectory dt fields exist."""
        result = TrajOptSolverResult(
            success=torch.tensor([True]),
        )
        assert hasattr(result, 'maximum_trajectory_dt')
        assert hasattr(result, 'minimum_trajectory_dt')


class TestTrajOptSolverResultClone:
    """Test TrajOptSolverResult.clone method."""

    def test_clone_creates_copy(self, sample_trajopt_result):
        """Test clone creates a new object."""
        cloned = sample_trajopt_result.clone()
        assert cloned is not sample_trajopt_result

    def test_clone_preserves_success(self, sample_trajopt_result):
        """Test clone preserves success tensor."""
        cloned = sample_trajopt_result.clone()
        assert torch.equal(cloned.success, sample_trajopt_result.success)

    def test_clone_preserves_solution(self, sample_trajopt_result):
        """Test clone preserves solution tensor."""
        cloned = sample_trajopt_result.clone()
        assert torch.equal(cloned.solution, sample_trajopt_result.solution)

    def test_clone_is_independent(self, sample_trajopt_result):
        """Test cloned tensors are independent."""
        cloned = sample_trajopt_result.clone()

        # Modify original
        sample_trajopt_result.success[0, 0] = False
        sample_trajopt_result.solution[0, 0, 0, 0] = 999.0

        # Clone should be unaffected
        assert cloned.success[0, 0] == True
        assert cloned.solution[0, 0, 0, 0] != 999.0

    def test_clone_with_interpolated_trajectory(self):
        """Test clone handles interpolated_trajectory correctly."""
        batch_size = 2
        num_seeds = 2
        dof = 7
        interp_horizon = 100

        interp_traj = JointState.from_position(
            torch.randn(batch_size, num_seeds, interp_horizon, dof)
        )

        result = TrajOptSolverResult(
            success=torch.tensor([[True, False], [True, True]]),
            interpolated_trajectory=interp_traj,
        )

        cloned = result.clone()
        assert cloned.interpolated_trajectory is not None

    def test_clone_with_interpolated_last_tstep(self):
        """Test clone handles interpolated_last_tstep correctly."""
        result = TrajOptSolverResult(
            success=torch.tensor([[True, False], [True, True]]),
            interpolated_last_tstep=torch.tensor([50, 60]),
        )

        cloned = result.clone()
        assert cloned.interpolated_last_tstep is not None
        assert torch.equal(cloned.interpolated_last_tstep, result.interpolated_last_tstep)


class TestTrajOptSolverResultInheritsFromBaseSolverResult:
    """Test TrajOptSolverResult inherits from BaseSolverResult."""

    def test_has_position_error(self, sample_trajopt_result):
        """Test has position_error from base class."""
        assert hasattr(sample_trajopt_result, 'position_error')

    def test_has_rotation_error(self, sample_trajopt_result):
        """Test has rotation_error from base class."""
        assert hasattr(sample_trajopt_result, 'rotation_error')

    def test_has_goalset_index(self, sample_trajopt_result):
        """Test has goalset_index from base class."""
        assert hasattr(sample_trajopt_result, 'goalset_index')

    def test_has_solve_time(self, sample_trajopt_result):
        """Test has solve_time from base class."""
        assert hasattr(sample_trajopt_result, 'solve_time')
        assert sample_trajopt_result.solve_time == 0.1

    def test_has_total_time(self, sample_trajopt_result):
        """Test has total_time from base class."""
        assert hasattr(sample_trajopt_result, 'total_time')
        assert sample_trajopt_result.total_time == 0.2

    def test_has_debug_info(self, sample_trajopt_result):
        """Test has debug_info from base class."""
        assert hasattr(sample_trajopt_result, 'debug_info')

    def test_has_optimized_seeds(self, sample_trajopt_result):
        """Test has optimized_seeds from base class."""
        assert hasattr(sample_trajopt_result, 'optimized_seeds')

    def test_has_metrics(self, sample_trajopt_result):
        """Test has metrics from base class."""
        assert hasattr(sample_trajopt_result, 'metrics')

    def test_has_batch_size(self, sample_trajopt_result):
        """Test has batch_size from base class."""
        assert hasattr(sample_trajopt_result, 'batch_size')
        assert sample_trajopt_result.batch_size == 2

    def test_has_num_seeds(self, sample_trajopt_result):
        """Test has num_seeds from base class."""
        assert hasattr(sample_trajopt_result, 'num_seeds')
        assert sample_trajopt_result.num_seeds == 2


class TestTrajOptSolverResultDeviceHandling:
    """Test TrajOptSolverResult device handling."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_tensors(self):
        """Test result with CUDA tensors."""
        result = TrajOptSolverResult(
            success=torch.tensor([True, False], device="cuda:0"),
            solution=torch.randn(2, 1, 32, 7, device="cuda:0"),
        )
        assert result.success.is_cuda
        assert result.solution.is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_clone_preserves_device(self):
        """Test clone preserves device."""
        result = TrajOptSolverResult(
            success=torch.tensor([True, False], device="cuda:0"),
            solution=torch.randn(2, 1, 32, 7, device="cuda:0"),
        )
        cloned = result.clone()
        assert cloned.success.is_cuda
        assert cloned.solution.is_cuda


class TestTrajOptSolverResultSolutionShape:
    """Test TrajOptSolverResult solution shapes."""

    def test_solution_shape_batch_seeds_horizon_dof(self):
        """Test solution shape is (batch, seeds, horizon, dof)."""
        batch_size = 4
        num_seeds = 2
        horizon = 32
        dof = 7

        result = TrajOptSolverResult(
            success=torch.ones(batch_size, num_seeds, dtype=torch.bool),
            solution=torch.randn(batch_size, num_seeds, horizon, dof),
        )

        assert result.solution.shape == (batch_size, num_seeds, horizon, dof)

    def test_js_solution_is_joint_state(self):
        """Test js_solution is JointState."""
        js_sol = JointState.from_position(torch.randn(2, 1, 32, 7))

        result = TrajOptSolverResult(
            success=torch.tensor([[True]]),
            js_solution=js_sol,
        )

        assert isinstance(result.js_solution, JointState)

