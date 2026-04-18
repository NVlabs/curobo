# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optimizer creation entry point mapping solver_type strings to optimizer classes.

Provides :func:`create_optimization_config` and :func:`create_optimizer` which
look up the solver_type in a registry and instantiate the corresponding config
and optimizer (L-BFGS, L-SR1, CG, MPPI, ES, etc.) with the given rollout list.
"""
from __future__ import annotations

from typing import Dict, List

from curobo._src.optim.external.scipy_opt import ScipyOpt, ScipyOptCfg
from curobo._src.optim.external.torch_opt import TorchOpt, TorchOptCfg
from curobo._src.optim.gradient.conjugate_gradient import (
    ConjugateGradientOpt,
    ConjugateGradientOptCfg,
)
from curobo._src.optim.gradient.gradient_descent import (
    GradientDescentOpt,
    GradientDescentOptCfg,
    LineSearchGradientDescentOpt,
)
from curobo._src.optim.gradient.lbfgs import LBFGSOpt, LBFGSOptCfg
from curobo._src.optim.gradient.lsr1 import LSR1Opt
from curobo._src.optim.particle.evolution_strategies import (
    EvolutionStrategies,
    EvolutionStrategiesCfg,
)
from curobo._src.optim.particle.mppi import MPPI, MPPICfg
from curobo._src.rollout.rollout_protocol import Rollout
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise


# Config mapping: solver_type string -> config class
_SOLVER_CONFIG_MAP = {
    "lbfgs": LBFGSOptCfg,
    "gradient_descent": GradientDescentOptCfg,
    "line_search_gradient_descent": GradientDescentOptCfg,
    "conjugate_gradient": ConjugateGradientOptCfg,
    "lsr1": LBFGSOptCfg,
    "scipy": ScipyOptCfg,
    "torch": TorchOptCfg,
    "mppi": MPPICfg,
    "es": EvolutionStrategiesCfg,
}

# Optimizer mapping: solver_type string -> optimizer class
_SOLVER_MAP = {
    "lbfgs": LBFGSOpt,
    "gradient_descent": GradientDescentOpt,
    "line_search_gradient_descent": LineSearchGradientDescentOpt,
    "conjugate_gradient": ConjugateGradientOpt,
    "lsr1": LSR1Opt,
    "scipy": ScipyOpt,
    "torch": TorchOpt,
    "mppi": MPPI,
    "es": EvolutionStrategies,
}


def create_optimization_config(
    config_dict: Dict,
    device_cfg: DeviceCfg,
):
    """Create an optimizer configuration based on solver_type in the config dictionary.

    Args:
        config_dict: Dictionary containing optimizer configuration including solver_type.
        device_cfg: Tensor device type information.

    Returns:
        The appropriate optimizer configuration object.
    """
    solver_type = config_dict.get("solver_type", "")
    config_class = _SOLVER_CONFIG_MAP.get(solver_type)
    if config_class is None:
        raise ValueError(
            f"Unsupported solver type: {solver_type}. "
            f"Available: {list(_SOLVER_CONFIG_MAP.keys())}"
        )
    config_dict = config_class.create_data_dict(config_dict.copy(), device_cfg)
    return config_class(**config_dict)


def create_optimizer(
    config,
    rollout: List[Rollout],
    use_cuda_graph: bool = False,
):
    """Create an optimizer instance based on the config object.

    All optimizers are standalone and accept use_cuda_graph as a constructor parameter.

    Args:
        config: Configuration object for the optimizer.
        rollout: List of rollout instances.
        use_cuda_graph: Whether to use CUDA graphs for acceleration.

    Returns:
        The appropriate optimizer instance.
    """
    solver_type = config.solver_type

    if not isinstance(rollout, list):
        log_and_raise("rollout must be a list of Rollout")

    optimizer_class = _SOLVER_MAP.get(solver_type)
    if optimizer_class is None:
        raise ValueError(
            f"Unsupported solver type: {solver_type}. "
            f"Available: {list(_SOLVER_MAP.keys())}"
        )

    return optimizer_class(
        config=config, rollout_list=rollout, use_cuda_graph=use_cuda_graph
    )
