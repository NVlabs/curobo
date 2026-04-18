# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rollout module.

Rollouts define the cost function and dynamics that an optimizer minimizes over. cuRobo's
built-in solvers use robot-specific rollouts internally; this public surface exposes
rollouts useful for writing standalone optimization examples and benchmarks.

Currently exposes:
    - :class:`RosenbrockRollout` / :class:`RosenbrockCfg`: A canonical non-convex test
      function, useful for validating custom optimizer configurations against a well-known
      problem.

Example:
    .. code-block:: python

        from curobo.rollout import RosenbrockRollout, RosenbrockCfg

        rollout = RosenbrockRollout(RosenbrockCfg(...))
"""
from curobo._src.rollout.rollout_rosenbrock import RosenbrockCfg, RosenbrockRollout

__all__ = [
    "RosenbrockCfg",
    "RosenbrockRollout",
]
