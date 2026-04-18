# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for BaseSolverResult.copy_at_batch_indices and
TrajOptSolverResult.copy_at_batch_indices."""

import pytest
import torch

from curobo._src.solver.solver_base_result import BaseSolverResult
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState


class TestBaseSolverResultCopyAtBatchIndices:
    """Test BaseSolverResult.copy_at_batch_indices."""

    def _make_result(self, batch_size=4, num_seeds=2, dof=7, fill=0.0):
        return BaseSolverResult(
            success=torch.zeros(batch_size, num_seeds, dtype=torch.bool),
            solution=torch.full((batch_size, num_seeds, 1, dof), fill),
            position_error=torch.full((batch_size, num_seeds), fill),
            rotation_error=torch.full((batch_size, num_seeds), fill),
            goalset_index=torch.full((batch_size, num_seeds), int(fill), dtype=torch.long),
            feasible=torch.zeros(batch_size, num_seeds, dtype=torch.bool),
            batch_size=batch_size,
            num_seeds=num_seeds,
        )

    def test_copies_masked_indices(self):
        dst = self._make_result(fill=0.0)
        src = self._make_result(fill=1.0)
        src.success[:] = True
        src.feasible[:] = True

        mask = torch.tensor([False, True, False, True])
        dst.copy_at_batch_indices(src, mask)

        assert dst.success[1].all()
        assert dst.success[3].all()
        assert not dst.success[0].any()
        assert not dst.success[2].any()

        assert (dst.solution[1] == 1.0).all()
        assert (dst.solution[3] == 1.0).all()
        assert (dst.solution[0] == 0.0).all()
        assert (dst.solution[2] == 0.0).all()

        assert (dst.position_error[1] == 1.0).all()
        assert (dst.position_error[0] == 0.0).all()

    def test_no_op_when_mask_all_false(self):
        dst = self._make_result(fill=0.0)
        src = self._make_result(fill=9.0)
        src.success[:] = True

        mask = torch.zeros(4, dtype=torch.bool)
        dst.copy_at_batch_indices(src, mask)

        assert not dst.success.any()
        assert (dst.solution == 0.0).all()

    def test_all_true_mask_copies_everything(self):
        dst = self._make_result(fill=0.0)
        src = self._make_result(fill=5.0)
        src.success[:] = True

        mask = torch.ones(4, dtype=torch.bool)
        dst.copy_at_batch_indices(src, mask)

        assert dst.success.all()
        assert (dst.solution == 5.0).all()

    def test_skips_none_fields(self):
        dst = BaseSolverResult(
            success=torch.zeros(4, 2, dtype=torch.bool),
            batch_size=4, num_seeds=2,
        )
        src = BaseSolverResult(
            success=torch.ones(4, 2, dtype=torch.bool),
            batch_size=4, num_seeds=2,
        )
        mask = torch.tensor([True, False, True, False])
        dst.copy_at_batch_indices(src, mask)

        assert dst.success[0].all()
        assert dst.success[2].all()
        assert not dst.success[1].any()

    def test_copies_js_solution(self):
        batch_size, num_seeds, dof = 4, 2, 7
        dst_pos = torch.zeros(batch_size, num_seeds, dof)
        src_pos = torch.ones(batch_size, num_seeds, dof)
        dst_vel = torch.zeros(batch_size, num_seeds, dof)
        src_vel = torch.ones(batch_size, num_seeds, dof) * 2.0

        dst = BaseSolverResult(
            success=torch.zeros(batch_size, num_seeds, dtype=torch.bool),
            js_solution=JointState(position=dst_pos, velocity=dst_vel),
            batch_size=batch_size, num_seeds=num_seeds,
        )
        src = BaseSolverResult(
            success=torch.ones(batch_size, num_seeds, dtype=torch.bool),
            js_solution=JointState(position=src_pos, velocity=src_vel),
            batch_size=batch_size, num_seeds=num_seeds,
        )

        mask = torch.tensor([False, False, True, True])
        dst.copy_at_batch_indices(src, mask)

        assert (dst.js_solution.position[2] == 1.0).all()
        assert (dst.js_solution.position[3] == 1.0).all()
        assert (dst.js_solution.position[0] == 0.0).all()
        assert (dst.js_solution.velocity[2] == 2.0).all()
        assert (dst.js_solution.velocity[0] == 0.0).all()


class TestTrajOptSolverResultCopyAtBatchIndices:
    """Test TrajOptSolverResult.copy_at_batch_indices handles interpolated trajectory."""

    def test_copies_interpolated_trajectory(self):
        batch_size, num_seeds, horizon, dof = 4, 1, 10, 7

        dst_interp = JointState(
            position=torch.zeros(batch_size, num_seeds, horizon, dof),
        )
        src_interp = JointState(
            position=torch.ones(batch_size, num_seeds, horizon, dof),
        )

        dst = TrajOptSolverResult(
            success=torch.zeros(batch_size, num_seeds, dtype=torch.bool),
            interpolated_trajectory=dst_interp,
            interpolated_last_tstep=torch.zeros(batch_size, num_seeds),
            batch_size=batch_size, num_seeds=num_seeds,
        )
        src = TrajOptSolverResult(
            success=torch.ones(batch_size, num_seeds, dtype=torch.bool),
            interpolated_trajectory=src_interp,
            interpolated_last_tstep=torch.ones(batch_size, num_seeds) * 5,
            batch_size=batch_size, num_seeds=num_seeds,
        )

        mask = torch.tensor([True, False, False, True])
        dst.copy_at_batch_indices(src, mask)

        assert (dst.interpolated_trajectory.position[0] == 1.0).all()
        assert (dst.interpolated_trajectory.position[3] == 1.0).all()
        assert (dst.interpolated_trajectory.position[1] == 0.0).all()
        assert dst.interpolated_last_tstep[0, 0] == 5.0
        assert dst.interpolated_last_tstep[1, 0] == 0.0

    def test_handles_none_interpolated(self):
        batch_size, num_seeds = 4, 1
        dst = TrajOptSolverResult(
            success=torch.zeros(batch_size, num_seeds, dtype=torch.bool),
            batch_size=batch_size, num_seeds=num_seeds,
        )
        src = TrajOptSolverResult(
            success=torch.ones(batch_size, num_seeds, dtype=torch.bool),
            batch_size=batch_size, num_seeds=num_seeds,
        )
        mask = torch.tensor([True, False, False, False])
        dst.copy_at_batch_indices(src, mask)
        assert dst.success[0].all()
