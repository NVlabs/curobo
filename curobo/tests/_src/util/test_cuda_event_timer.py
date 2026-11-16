# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CUDA-event interval checkpoints."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src import runtime
from curobo._src.util.cuda_event_timer import CudaEventTimer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_event_timer_accumulates_named_checkpoint_intervals():
    """Named adjacent intervals telescope to the original total event span."""
    value = torch.ones(1 << 18, device="cuda")
    torch.cuda.synchronize()
    timer = CudaEventTimer().start()
    value.add_(1.0)
    timer.checkpoint("repeated")
    value.mul_(2.0)
    timer.checkpoint("repeated")
    value.add_(3.0)
    total = timer.stop(final_interval="final")

    assert set(timer.intervals) == {"repeated", "final"}
    assert timer.intervals["repeated"] >= 0.0
    assert timer.intervals["final"] >= 0.0
    assert sum(timer.intervals.values()) == pytest.approx(total, abs=2.0e-6)


def test_cuda_event_timer_checkpoints_are_noops_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    """Respect the global timer switch without allocating CUDA events."""
    monkeypatch.setattr(runtime, "cuda_event_timers", False)
    timer = CudaEventTimer().start().checkpoint("phase")

    assert timer.stop(final_interval="final") == 0.0
    assert timer.intervals == {}
