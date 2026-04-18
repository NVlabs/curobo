# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from curobo._src.types.sequence_tool_pose import SequenceGoalToolPose


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    if request.param.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


def _make_seq(num_frames, num_envs, num_links, device, num_goalset=1):
    """Helper to create a valid SequenceGoalToolPose (5D tensors)."""
    tool_frames = [f"link_{i}" for i in range(num_links)]
    position = torch.rand(num_frames, num_envs, num_links, num_goalset, 3, device=device)
    quat = torch.rand(num_frames, num_envs, num_links, num_goalset, 4, device=device)
    quat[..., 0] = 1.0
    return SequenceGoalToolPose(tool_frames=tool_frames, position=position, quaternion=quat)


class TestSequenceGoalToolPose:

    def test_properties(self, device):
        seq = _make_seq(10, 2, 3, device)
        assert seq.num_frames == 10
        assert seq.num_envs == 2
        assert seq.num_links == 3
        assert seq.num_goalset == 1
        assert seq.device == device

    def test_get_frame_shape(self, device):
        seq = _make_seq(8, 2, 4, device)
        tp = seq.get_frame(3)
        assert tp.position.shape == (2, 1, 4, 1, 3)
        assert tp.quaternion.shape == (2, 1, 4, 1, 4)
        assert tp.horizon == 1
        assert tp.num_goalset == 1
        assert tp.tool_frames == seq.tool_frames

    def test_get_frame_values(self, device):
        """get_frame should return the correct slice of data."""
        seq = _make_seq(5, 1, 2, device)
        tp = seq.get_frame(2)
        expected_pos = seq.position[2].unsqueeze(1)
        expected_quat = seq.quaternion[2].unsqueeze(1)
        assert torch.allclose(tp.position, expected_pos)
        assert torch.allclose(tp.quaternion, expected_quat)

    def test_get_frame_boundary(self, device):
        """First and last frame should be accessible."""
        seq = _make_seq(6, 1, 2, device)
        tp_first = seq.get_frame(0)
        tp_last = seq.get_frame(5)
        assert tp_first.position.shape == (1, 1, 2, 1, 3)
        assert tp_last.position.shape == (1, 1, 2, 1, 3)

    def test_clone(self, device):
        seq = _make_seq(4, 1, 2, device)
        cloned = seq.clone()
        assert torch.allclose(cloned.position, seq.position)
        assert torch.allclose(cloned.quaternion, seq.quaternion)
        assert cloned.tool_frames == seq.tool_frames
        cloned.position[0, 0, 0, 0, 0] = 999.0
        assert seq.position[0, 0, 0, 0, 0] != 999.0

    def test_invalid_position_ndim_raises(self, device):
        with pytest.raises(ValueError, match="5D"):
            SequenceGoalToolPose(
                tool_frames=["a"],
                position=torch.zeros(3, 3, device=device),
                quaternion=torch.zeros(3, 1, 1, 1, 4, device=device),
            )

    def test_invalid_quaternion_ndim_raises(self, device):
        with pytest.raises(ValueError, match="5D"):
            SequenceGoalToolPose(
                tool_frames=["a"],
                position=torch.zeros(3, 1, 1, 1, 3, device=device),
                quaternion=torch.zeros(3, 4, device=device),
            )

    def test_n_links_mismatch_raises(self, device):
        with pytest.raises(ValueError):
            SequenceGoalToolPose(
                tool_frames=["a", "b"],
                position=torch.zeros(3, 1, 3, 1, 3, device=device),
                quaternion=torch.zeros(3, 1, 3, 1, 4, device=device),
            )

    def test_single_frame_single_env(self, device):
        seq = _make_seq(1, 1, 1, device)
        assert seq.num_frames == 1
        tp = seq.get_frame(0)
        assert tp.position.shape == (1, 1, 1, 1, 3)
