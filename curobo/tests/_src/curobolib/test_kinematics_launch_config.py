#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CUDA-core kinematics launch configuration."""

from __future__ import annotations

import pytest

import curobo._src.runtime as curobo_runtime
from curobo._src.curobolib.backends.cuda_core_backend.kinematics import (
    _select_kinematics_backward_threads,
)
from curobo._src.curobolib.backends.cuda_core_backend.kinematics_config import (
    KinematicsLaunchCfg,
)


def test_jacobian_vjp_launch_uses_four_32_lane_batches_per_block() -> None:
    config = KinematicsLaunchCfg.calculate_jacobian_vjp_config(
        batch_size=256, num_links=13
    )

    assert config.grid == (64, 1, 1)
    assert config.block == (128, 1, 1)
    assert config.shmem_size == 4 * 13 * 12 * 4


@pytest.mark.parametrize(
    ("sm_count", "expected_threads"),
    [(16, 64), (32, 64), (33, 128), (128, 128)],
)
def test_kinematics_backward_threads_follow_gpu_policy(
    sm_count: int, expected_threads: int
) -> None:
    assert _select_kinematics_backward_threads(sm_count) == expected_threads


def test_kernel_launch_sm_count_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(curobo_runtime, "kernel_launch_sm_count", 32)
    assert curobo_runtime.resolve_kernel_launch_sm_count(120) == 32


def test_kernel_launch_sm_count_defaults_to_device() -> None:
    assert curobo_runtime.resolve_kernel_launch_sm_count(120) == 120


@pytest.mark.parametrize("invalid_value", [0, -1, True, 32.0])
def test_kernel_launch_sm_count_rejects_invalid_override(
    monkeypatch: pytest.MonkeyPatch, invalid_value: object
) -> None:
    monkeypatch.setattr(curobo_runtime, "kernel_launch_sm_count", invalid_value)
    with pytest.raises(ValueError, match="positive integer or None"):
        curobo_runtime.resolve_kernel_launch_sm_count(120)
