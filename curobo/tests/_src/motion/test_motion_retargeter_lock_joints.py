# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MotionRetargeter on a robot with non-empty lock_joints.

cuRobo v0.8.0 / origin/main bug: `MotionRetargeter._solve_global_ik` and
`_solve_local_ik` both `.view(num_envs, action_dim)` on
`js_solution.position` / `.velocity`, but the internal IK preserves
locked-joint slots in those tensors. The view fails with RuntimeError
whenever `lock_joints` is non-empty.

The fix routes both sites through `Kinematics.get_active_js` to drop the
locked-joint columns before the view.
"""
import pytest
import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.motion.motion_retargeter import MotionRetargeter
from curobo._src.motion.motion_retargeter_cfg import MotionRetargeterCfg
from curobo._src.motion.motion_retargeter_result import RetargetResult
from curobo._src.types.sequence_tool_pose import SequenceGoalToolPose

# simple_mimic_robot.yml ships with lock_joints={chain_1_active_joint_1: 0.2}
# and cspace.joint_names=[chain_1_active_joint_1, active_joint_2]. The locked
# joint is in cspace, exactly the shape that can trigger any regression.
ROBOT_CFG = "simple_mimic_robot.yml"
TOOL_FRAMES = ["ee_link"]
LOCKED_JOINT = "chain_1_active_joint_1"


def _criteria():
    return {
        name: ToolPoseCriteria.track_position_and_orientation(
            xyz=[1.0, 1.0, 1.0], rpy=[0.5, 0.5, 0.5],
        )
        for name in TOOL_FRAMES
    }


def _make_sequence(num_frames, num_envs, num_links, device):
    # Mirrors sibling test_motion_retargeter.py's signature for style consistency.
    position = torch.zeros(num_frames, num_envs, num_links, 1, 3, device=device)
    position[..., 2] = 0.5
    quaternion = torch.zeros(num_frames, num_envs, num_links, 1, 4, device=device)
    quaternion[..., 0] = 1.0
    return SequenceGoalToolPose(
        tool_frames=TOOL_FRAMES,
        position=position,
        quaternion=quaternion,
    )


@pytest.fixture(scope="module")
def retargeter_ik_locked():
    """MotionRetargeter in IK mode on simple_mimic_robot.yml (non-empty lock_joints)."""
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


class TestMotionRetargeterLockJoints:

    def test_first_call_no_runtime_error(self, retargeter_ik_locked):
        """_solve_global_ik must not RuntimeError on cspace with locked joints."""
        retargeter_ik_locked.reset()
        seq = _make_sequence(1, 1, len(TOOL_FRAMES), "cuda:0")
        result = retargeter_ik_locked.solve_frame(seq.get_frame(0))
        assert isinstance(result, RetargetResult)

    def test_subsequent_call_no_runtime_error(self, retargeter_ik_locked):
        """_solve_local_ik (frame 2+, position view) must not RuntimeError."""
        retargeter_ik_locked.reset()
        seq = _make_sequence(2, 1, len(TOOL_FRAMES), "cuda:0")
        retargeter_ik_locked.solve_frame(seq.get_frame(0))
        result = retargeter_ik_locked.solve_frame(seq.get_frame(1))
        assert isinstance(result, RetargetResult)

    def test_output_contract_action_dim(self, retargeter_ik_locked):
        """Post-fix contract: output JointState is action-dim only (locked joints excluded)."""
        retargeter_ik_locked.reset()
        seq = _make_sequence(1, 1, len(TOOL_FRAMES), "cuda:0")
        result = retargeter_ik_locked.solve_frame(seq.get_frame(0))
        action_dim = retargeter_ik_locked.num_dof
        assert result.joint_state.position.shape == (1, action_dim)
        assert list(result.joint_state.joint_names) == retargeter_ik_locked.joint_names

    def test_locked_joints_excluded_from_output(self, retargeter_ik_locked):
        """Locked joint must NOT appear in the returned joint_state."""
        retargeter_ik_locked.reset()
        seq = _make_sequence(1, 1, len(TOOL_FRAMES), "cuda:0")
        result = retargeter_ik_locked.solve_frame(seq.get_frame(0))
        assert LOCKED_JOINT not in result.joint_state.joint_names, (
            f"locked joint leaked into output: {LOCKED_JOINT}"
        )

    def test_solve_sequence_no_runtime_error(self, retargeter_ik_locked):
        """End-to-end: 5-frame batch on lock_joints cspace."""
        retargeter_ik_locked.reset()
        num_frames = 5
        seq = _make_sequence(num_frames, 1, len(TOOL_FRAMES), "cuda:0")
        result = retargeter_ik_locked.solve_sequence(seq)
        assert result.joint_state.position.shape == (1, num_frames, retargeter_ik_locked.num_dof)
