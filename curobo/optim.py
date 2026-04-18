# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optimizer module.

Public surface for cuRobo's underlying optimizers. Most users should prefer the higher-level
solvers (:class:`~curobo.inverse_kinematics.InverseKinematics`,
:class:`~curobo.trajectory_optimizer.TrajectoryOptimizer`,
:class:`~curobo.motion_planner.MotionPlanner`,
:class:`~curobo.model_predictive_control.ModelPredictiveControl`) which wrap these
optimizers with sensible defaults.

Use this module when building custom optimization pipelines, for example to minimize a
user-defined rollout with MPPI, evolution strategies, L-BFGS, PyTorch, or SciPy optimizers,
or to chain them with :class:`MultiStageOptimizer`.

Example:
    .. code-block:: python

        from curobo.optim import MPPI, MPPICfg

        cfg = MPPICfg(...)
        opt = MPPI(cfg)
        result = opt.optimize(rollout, seed=x0)
"""
from curobo._src.optim.external.scipy_opt import ScipyOpt, ScipyOptCfg
from curobo._src.optim.external.torch_opt import TorchOpt, TorchOptCfg
from curobo._src.optim.gradient.lbfgs import LBFGSOpt, LBFGSOptCfg
from curobo._src.optim.multi_stage_optimizer import MultiStageOptimizer
from curobo._src.optim.particle.evolution_strategies import (
    EvolutionStrategies,
    EvolutionStrategiesCfg,
)
from curobo._src.optim.particle.mppi import MPPI, MPPICfg

__all__ = [
    "EvolutionStrategies",
    "EvolutionStrategiesCfg",
    "LBFGSOpt",
    "LBFGSOptCfg",
    "MPPI",
    "MPPICfg",
    "MultiStageOptimizer",
    "ScipyOpt",
    "ScipyOptCfg",
    "TorchOpt",
    "TorchOptCfg",
]
