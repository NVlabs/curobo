# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity, state lifecycle and dispatch tests for bounded vector-free L-BFGS."""

from __future__ import annotations

import pytest
import torch

from curobo._src.curobolib.backends.cuda_core_backend.optimization_config import (
    LBFGSVectorFreeLaunchCfg,
)
from curobo._src.curobolib.cuda_ops.optimization import LBFGScu
from curobo._src.optim.components.quasi_newton_buffers import QuasiNewtonBuffers
from curobo._src.types.device_cfg import DeviceCfg


@pytest.mark.parametrize(
    "history,dimension",
    [(1, 1), (3, 32), (8, 7), (15, 112), (16, 129), (26, 128), (26, 129), (30, 256)],
)
@pytest.mark.parametrize("geometric", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_vector_free_history_and_candidates(
    history: int, dimension: int, geometric: bool, shared: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve history, retry behavior, scaling and arbitrary candidate counts."""
    monkeypatch.setattr(
        LBFGSVectorFreeLaunchCfg,
        "specialization",
        staticmethod(lambda d, h, arch=None: (1 if d <= 128 else 2, shared)),
    )
    torch.manual_seed(9012)
    batch = 4
    reference = QuasiNewtonBuffers(DeviceCfg(), history)
    actual = QuasiNewtonBuffers(DeviceCfg(), history)
    reference.resize(batch, dimension)
    actual.resize(batch, dimension, use_vector_free_operator=True)
    steps = [torch.zeros(batch, dimension, device="cuda") for _ in range(2)]
    candidates = [torch.zeros(batch, 5, dimension, device="cuda") for _ in range(2)]
    scales = torch.tensor([0.01, 0.1, 0.3, 0.5, 1.0], device="cuda")
    beta = torch.tensor([1.0, 0.98, 0.5, 0.1], device="cuda")
    maximum = torch.full((dimension,), 0.25, device="cuda")
    position = torch.zeros(batch, dimension, device="cuda")
    weights = torch.linspace(0.5, 3.0, dimension, device="cuda")
    for iteration in range(2 * history + 5):
        position.add_(torch.randn_like(position) * 0.1)
        gradient = position * weights
        allowed = torch.ones(batch, dtype=torch.bool, device="cuda")
        status = torch.zeros(batch, dtype=torch.int32, device="cuda")
        if iteration % 7 == 3:
            allowed[0] = False
        if iteration % 11 == 4:
            status[1] = 5
        if iteration % 13 == 5:
            gradient[2] = actual.grad_0[2, :, 0]
        if iteration % 17 == 6:
            gradient[3] = actual.grad_0[3, :, 0] - (position[3] - actual.x_0[3, :, 0])
        for index, buffers in enumerate((reference, actual)):
            LBFGScu.apply(
                steps[index],
                candidates[index],
                scales,
                beta,
                buffers.rho,
                buffers.y,
                buffers.s,
                position,
                gradient[:, None, :],
                buffers.x_0,
                buffers.grad_0,
                buffers.history_head,
                buffers.history_count,
                0.023,
                True,
                True,
                geometric,
                maximum,
                True,
                allowed,
                status,
                buffers.cross_products,
            )
        torch.testing.assert_close(steps[1], steps[0], rtol=3e-4, atol=3e-5)
        torch.testing.assert_close(candidates[1], candidates[0], rtol=3e-4, atol=3e-5)
        for field in ("rho", "s", "y", "history_head", "history_count", "x_0", "grad_0"):
            torch.testing.assert_close(
                getattr(actual, field), getattr(reference, field), rtol=3e-4, atol=3e-5
            )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        LBFGScu.apply(
            steps[1],
            candidates[1],
            scales,
            beta,
            actual.rho,
            actual.y,
            actual.s,
            position,
            gradient[:, None, :],
            actual.x_0,
            actual.grad_0,
            actual.history_head,
            actual.history_count,
            0.023,
            True,
            True,
            geometric,
            maximum,
            True,
            allowed,
            status,
            actual.cross_products,
        )
    graph.replay()
    torch.cuda.synchronize()


def test_vector_free_reset_and_mpc_shift() -> None:
    """Rebuild products after shifting vectors and clear only selected problems."""
    buffers = QuasiNewtonBuffers(DeviceCfg(), 8)
    buffers.resize(4, 32, use_vector_free_operator=True)
    buffers.s.normal_()
    buffers.y.copy_(buffers.s * 2)
    buffers.history_count.fill_(8)
    buffers.history_head.fill_(3)
    buffers.shift(1, 4)
    buffers.rebuild_vector_free_state()
    s = buffers.s[..., 0].transpose(0, 1)
    y = buffers.y[..., 0].transpose(0, 1)
    expected = (s.unsqueeze(2) * y.unsqueeze(1)).sum(-1)
    torch.testing.assert_close(buffers.cross_products, expected)
    torch.testing.assert_close(
        buffers.rho[..., 0, 0].T, expected.diagonal(dim1=1, dim2=2).reciprocal()
    )
    retained = buffers.cross_products[1:].clone()
    buffers.clear(torch.tensor([True, False, False, False], device="cuda"))
    assert torch.count_nonzero(buffers.cross_products[0]) == 0
    torch.testing.assert_close(buffers.cross_products[1:], retained)
