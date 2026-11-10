# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock

import pytest
import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.motion.motion_retargeter import MotionRetargeter
from curobo._src.motion.motion_retargeter_cfg import MotionRetargeterCfg
from curobo._src.motion.motion_retargeter_result import RetargetResult
from curobo._src.solver.solver_ik_result import IKSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.sequence_tool_pose import SequenceGoalToolPose

ROBOT_CFG = "unitree_g1_29dof_retarget.yml"

TOOL_FRAMES = [
    "pelvis",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
]


def _criteria():
    return {
        name: ToolPoseCriteria.track_pose(
            xyz=[1.0, 1.0, 1.0], rpy=[0.5, 0.5, 0.5],
        )
        for name in TOOL_FRAMES
    }


def _make_sequence(num_frames, num_envs, num_links, device):
    """Create a SequenceGoalToolPose with constant identity targets at the origin."""
    position = torch.zeros(num_frames, num_envs, num_links, 1, 3, device=device)
    position[..., 2] = 0.75
    quaternion = torch.zeros(num_frames, num_envs, num_links, 1, 4, device=device)
    quaternion[..., 0] = 1.0
    return SequenceGoalToolPose(
        tool_frames=TOOL_FRAMES,
        position=position,
        quaternion=quaternion,
    )


@pytest.fixture(scope="module")
def retargeter_ik():
    """Build a MotionRetargeter in IK mode (module-scoped to amortize init)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for MotionRetargeter")
    cfg = MotionRetargeterCfg.create(
        robot=ROBOT_CFG,
        tool_pose_criteria=_criteria(),
        num_envs=1,
        use_mpc=False,
        self_collision_check=False,
    )
    return MotionRetargeter(cfg)


class TestMotionRetargeterIK:

    def test_solve_sequence_tracks_reachable_motion(
        self, retargeter_ik: MotionRetargeter
    ) -> None:
        """Retargeting follows FK-generated moving targets instead of returning zero poses."""
        reference = retargeter_ik.default_joint_state.position.view(1, -1).repeat(3, 1)
        reference[:, retargeter_ik.joint_names.index("base_j_z")] = torch.tensor(
            [0.8, 0.81, 0.82], device=reference.device, dtype=reference.dtype
        )
        targets = retargeter_ik.kinematics.compute_kinematics(
            JointState.from_position(reference, joint_names=retargeter_ik.joint_names)
        ).tool_poses
        sequence = SequenceGoalToolPose(
            tool_frames=targets.tool_frames,
            position=targets.position.unsqueeze(-2).clone(),
            quaternion=targets.quaternion.unsqueeze(-2).clone(),
        )
        result = retargeter_ik.solve_sequence(sequence)
        achieved = retargeter_ik.kinematics.compute_kinematics(result.joint_state).tool_poses
        error = (achieved.position - targets.position.transpose(0, 1)).norm(dim=-1)
        assert error.max() < 0.01
        assert torch.isfinite(result.joint_state.position).all()
        limits = retargeter_ik.kinematics.get_joint_limits().position
        assert (result.joint_state.position >= limits[0] - 1e-3).all()
        assert (result.joint_state.position <= limits[1] + 1e-3).all()

    def test_infeasible_initial_ik_raises(
        self, retargeter_ik: MotionRetargeter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An infeasible initial solve cannot initialize a sequence or reach export."""
        retargeter_ik.reset()
        feasible = torch.zeros((1, 1), device="cuda:0", dtype=torch.bool)
        monkeypatch.setattr(
            retargeter_ik._global_ik_solver,
            "solve_pose",
            Mock(return_value=IKSolverResult(success=feasible, feasible=feasible)),
        )
        target = _make_sequence(1, 1, len(TOOL_FRAMES), "cuda:0").get_frame(0)
        with pytest.raises(ValueError, match="initial IK failed"):
            retargeter_ik.solve_frame(target)
        assert retargeter_ik._prev_solution is None

    def test_infeasible_local_ik_holds_only_failed_environments(
        self, retargeter_ik: MotionRetargeter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed environment holds its previous pose and zero velocity independently."""
        previous = retargeter_ik.default_joint_state.position.view(1, -1).repeat(2, 1)
        candidate = previous.clone()
        candidate[0, 0] += 0.01
        candidate[1] += 1000.0
        feasible = torch.tensor([[True], [False]], device=previous.device)
        joint_state = JointState.from_position(
            candidate.unsqueeze(1), joint_names=retargeter_ik.joint_names
        )
        joint_state.velocity = torch.ones_like(joint_state.position)
        monkeypatch.setattr(retargeter_ik, "_num_envs", 2)
        monkeypatch.setattr(retargeter_ik, "_prev_solution", previous.clone())
        monkeypatch.setattr(
            retargeter_ik._local_ik_solver,
            "solve_pose",
            Mock(return_value=IKSolverResult(
                success=feasible, feasible=feasible, js_solution=joint_state
            )),
        )
        target = _make_sequence(1, 2, len(TOOL_FRAMES), "cuda:0").get_frame(0)
        result = retargeter_ik.solve_frame(target)
        torch.testing.assert_close(result.joint_state.position[0], candidate[0])
        torch.testing.assert_close(result.joint_state.position[1], previous[1])
        torch.testing.assert_close(retargeter_ik._prev_velocity[0], torch.ones_like(previous[0]))
        torch.testing.assert_close(retargeter_ik._prev_velocity[1], torch.zeros_like(previous[1]))

    def test_local_ik_validates_bounded_candidate(
        self, retargeter_ik: MotionRetargeter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repairing a bound violation still requires a second collision/constraint check."""
        retargeter_ik.reset()
        previous = retargeter_ik.default_joint_state.position.view(1, -1).clone()
        monkeypatch.setattr(retargeter_ik, "_prev_solution", previous.clone())
        candidate = previous.clone()
        candidate[:, 0] = 1000.0
        invalid_js = JointState.from_position(
            candidate.unsqueeze(1), joint_names=retargeter_ik.joint_names
        )
        repaired_js = JointState.from_position(
            previous.unsqueeze(1), joint_names=retargeter_ik.joint_names
        )
        feasible = torch.ones((1, 1), device=previous.device, dtype=torch.bool)
        solve = Mock(side_effect=[
            IKSolverResult(success=~feasible, feasible=~feasible, js_solution=invalid_js),
            IKSolverResult(success=feasible, feasible=feasible, js_solution=repaired_js),
        ])
        monkeypatch.setattr(retargeter_ik._local_ik_solver, "solve_pose", solve)
        target = _make_sequence(1, 1, len(TOOL_FRAMES), "cuda:0").get_frame(0)
        result = retargeter_ik.solve_frame(target)
        assert solve.call_count == 2
        validation = solve.call_args.kwargs
        assert validation["run_optimizer"] is False
        bounded = validation["seed_config"].view_as(previous)
        limits = retargeter_ik.kinematics.get_joint_limits()
        assert (bounded >= limits.position[0]).all()
        assert (bounded <= limits.position[1]).all()
        assert (bounded - previous <= limits.velocity[1] * retargeter_ik.config.optimization_dt).all()
        torch.testing.assert_close(result.joint_state.position, previous)

    def test_joint_names(self, retargeter_ik):
        assert isinstance(retargeter_ik.joint_names, list)
        assert len(retargeter_ik.joint_names) == retargeter_ik.num_dof
        assert retargeter_ik.num_dof > 0

    def test_solve_frame_first_call(self, retargeter_ik):
        """First call should use global IK and return a result."""
        retargeter_ik.reset()
        seq = _make_sequence(1, 1, len(TOOL_FRAMES), "cuda:0")
        tp = seq.get_frame(0)
        result = retargeter_ik.solve_frame(tp)
        assert isinstance(result, RetargetResult)
        assert result.joint_state.position.shape == (1, retargeter_ik.num_dof)
        assert result.trajectory is None

    def test_solve_frame_subsequent_call(self, retargeter_ik):
        """Second call should use local IK (warm-started)."""
        retargeter_ik.reset()
        seq = _make_sequence(2, 1, len(TOOL_FRAMES), "cuda:0")
        retargeter_ik.solve_frame(seq.get_frame(0))
        result = retargeter_ik.solve_frame(seq.get_frame(1))
        assert result.joint_state.position.shape == (1, retargeter_ik.num_dof)
        assert result.trajectory is None

    def test_solve_sequence(self, retargeter_ik):
        num_frames = 5
        seq = _make_sequence(num_frames, 1, len(TOOL_FRAMES), "cuda:0")
        result = retargeter_ik.solve_sequence(seq)
        assert result.joint_state.position.shape == (1, num_frames, retargeter_ik.num_dof)
        assert result.trajectory is None

    def test_solve_sequence_resets_state(self, retargeter_ik):
        """solve_sequence should call reset internally so each call starts fresh."""
        seq = _make_sequence(3, 1, len(TOOL_FRAMES), "cuda:0")
        retargeter_ik.solve_sequence(seq)
        assert retargeter_ik._prev_solution is not None
        retargeter_ik.solve_sequence(seq)
        assert retargeter_ik._prev_solution is not None

    def test_batch_size_mismatch_raises(self, retargeter_ik):
        """GoalToolPose with wrong num_envs should raise."""
        retargeter_ik.reset()
        seq = _make_sequence(1, 2, len(TOOL_FRAMES), "cuda:0")
        with pytest.raises(Exception):
            retargeter_ik.solve_frame(seq.get_frame(0))

    def test_sequence_batch_mismatch_raises(self, retargeter_ik):
        """SequenceGoalToolPose with wrong num_envs should raise."""
        seq = _make_sequence(3, 2, len(TOOL_FRAMES), "cuda:0")
        with pytest.raises(Exception):
            retargeter_ik.solve_sequence(seq)

    def test_reset_clears_state(self, retargeter_ik):
        """After reset, next solve_frame should use global IK again."""
        seq = _make_sequence(2, 1, len(TOOL_FRAMES), "cuda:0")
        retargeter_ik.reset()
        retargeter_ik.solve_frame(seq.get_frame(0))
        retargeter_ik.reset()
        assert retargeter_ik._prev_solution is None


class TestMotionRetargeterMPC:

    @pytest.fixture(scope="class")
    def retargeter_mpc(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for MotionRetargeter")
        cfg = MotionRetargeterCfg.create(
            robot=ROBOT_CFG,
            tool_pose_criteria=_criteria(),
            num_envs=1,
            use_mpc=True,
            steps_per_target=2,
            self_collision_check=False,
        )
        return MotionRetargeter(cfg)

    def test_solve_frame_mpc_first_frame(self, retargeter_mpc):
        """First frame should still use global IK even in MPC mode."""
        retargeter_mpc.reset()
        seq = _make_sequence(1, 1, len(TOOL_FRAMES), "cuda:0")
        result = retargeter_mpc.solve_frame(seq.get_frame(0))
        assert result.joint_state.position.shape == (1, retargeter_mpc.num_dof)
        assert result.trajectory is None

    def test_solve_frame_mpc_returns_trajectory(self, retargeter_mpc):
        """MPC subsequent frames should return a trajectory."""
        retargeter_mpc.reset()
        seq = _make_sequence(2, 1, len(TOOL_FRAMES), "cuda:0")
        retargeter_mpc.solve_frame(seq.get_frame(0))
        result = retargeter_mpc.solve_frame(seq.get_frame(1))
        assert result.joint_state.position.shape == (1, retargeter_mpc.num_dof)
        assert result.trajectory is not None
        assert result.trajectory.position.ndim == 3

    def test_solve_sequence_mpc(self, retargeter_mpc):
        num_frames = 4
        seq = _make_sequence(num_frames, 1, len(TOOL_FRAMES), "cuda:0")
        result = retargeter_mpc.solve_sequence(seq)
        assert result.joint_state.position.shape == (
            1, num_frames, retargeter_mpc.num_dof,
        )
        assert result.trajectory is not None
        assert result.trajectory.position.shape[0] == 1
        assert result.trajectory.position.shape[2] == retargeter_mpc.num_dof


class TestRetargetResult:

    def test_ik_result_no_trajectory(self):
        from curobo._src.state.state_joint import JointState
        joint_state = JointState.from_position(torch.zeros(1, 10))
        result = RetargetResult(joint_state=joint_state)
        assert result.trajectory is None

    def test_mpc_result_with_trajectory(self):
        from curobo._src.state.state_joint import JointState
        joint_state = JointState.from_position(torch.zeros(1, 10))
        traj = JointState.from_position(torch.zeros(1, 20, 10))
        result = RetargetResult(joint_state=joint_state, trajectory=traj)
        assert result.trajectory is not None
        assert result.trajectory.position.shape == (1, 20, 10)
