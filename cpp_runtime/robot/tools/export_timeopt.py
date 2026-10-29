#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export robot-independent timeopt tables using the Python solver's setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from curobo._src.trajectory.timeopt.timeopt_basis import CubicTimingBasis, TimeOptProblem
from curobo._src.trajectory.timeopt.timeopt_solver import CudaTimeOptSolver


def export_timeopt_tables(
    directory: Path, dof: int, knots: int, reference: bool, intervals: int = 511
) -> Path:
    """Export static float32/int32 tensors; no GPU or runtime model is required."""
    basis = CubicTimingBasis(32, torch.device("cpu"))
    spans = knots + 2 + reference
    internal = np.arange(1, spans) / spans
    uniform = np.linspace(0, 1, intervals + 1)
    grid = np.r_[uniform, internal, internal, basis.breaks_cpu[1:-1]]
    segments = np.minimum(np.floor(grid * spans).astype(np.int32), spans - 1)
    local = grid * spans - segments
    offset = intervals + 1
    segments[offset : offset + spans - 1] = np.arange(spans - 1)
    local[offset : offset + spans - 1] = 1.0
    segments[offset + spans - 1 : offset + 2 * (spans - 1)] = np.arange(1, spans)
    local[offset + spans - 1 : offset + 2 * (spans - 1)] = 0.0
    path = torch.empty((4, 1, len(grid), dof), dtype=torch.float32)
    problem = TimeOptProblem(
        basis,
        grid,
        path,
        torch.empty((1, 3, len(grid), dof)),
        tuple(torch.ones(dof) for _ in range(5)),
        intervals,
    )
    solver = CudaTimeOptSolver(problem)
    nodes, weights = np.polynomial.legendre.leggauss(6)
    quadrature = uniform[:-1, None] + np.diff(uniform)[:, None] * 0.5 * (nodes + 1)
    solver.prepare_output(basis.at(quadrature.ravel())[0], basis.tensor(0.5 * weights), intervals)
    tensors = {
        name: getattr(solver, name)
        for name in (
            "basis",
            "point_first",
            "meta",
            "offsets",
            "starts",
            "time_basis",
            "time_first",
            "time_weights",
        )
    }
    tensors.update(
        segments=torch.as_tensor(segments, dtype=torch.int32),
        local=basis.tensor(local),
        polynomial=basis.polynomial,
        nodes=basis.tensor(0.5 * (nodes + 1)),
    )
    prefix = f"timeopt_{knots}_{int(reference)}"
    directory.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        schema_version=3,
        num_dof=dof,
        num_knots=knots,
        reference=int(reference),
        intervals=intervals,
        points=len(grid),
        tiles=solver.tiles,
        tensors={},
    )
    for name, tensor in tensors.items():
        array = tensor.contiguous().numpy()
        assert array.dtype in (np.dtype("float32"), np.dtype("int32"))
        filename = f"{prefix}_{name}.bin"
        array.tofile(directory / filename)
        metadata["tensors"][name] = dict(file=filename, dtype=str(array.dtype), bytes=array.nbytes)
    manifest = directory / f"{prefix}.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--dof", type=int, required=True)
    parser.add_argument("--knots", type=int, nargs="+", default=[12, 16, 32, 64])
    args = parser.parse_args()
    for knots in args.knots:
        for reference in (False, True):
            export_timeopt_tables(args.directory, args.dof, knots, reference)
