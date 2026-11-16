# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optimizer creation entry point mapping solver_type strings to optimizer classes.

Provides :func:`create_optimization_config` and :func:`create_optimizer` which
look up the solver_type in a registry and instantiate the corresponding config
and optimizer (L-BFGS, CG, MPPI, ES, etc.) with the given rollout list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Type

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
from curobo._src.optim.particle.evolution_strategies import (
    EvolutionStrategies,
    EvolutionStrategiesCfg,
)
from curobo._src.optim.particle.mppi import MPPI, MPPICfg
from curobo._src.rollout.rollout_protocol import Rollout
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise


@dataclass(frozen=True)
class _OptimizerSpec:
    """Pair an optimizer configuration with its implementation class."""

    config_class: Type[Any]
    """Configuration class associated with the solver type."""

    optimizer_class: Type[Any]
    """Optimizer implementation associated with the solver type."""


_SOLVER_REGISTRY: Dict[str, _OptimizerSpec] = {
    "lbfgs": _OptimizerSpec(LBFGSOptCfg, LBFGSOpt),
    "gradient_descent": _OptimizerSpec(GradientDescentOptCfg, GradientDescentOpt),
    "line_search_gradient_descent": _OptimizerSpec(
        GradientDescentOptCfg, LineSearchGradientDescentOpt
    ),
    "conjugate_gradient": _OptimizerSpec(ConjugateGradientOptCfg, ConjugateGradientOpt),
    "scipy": _OptimizerSpec(ScipyOptCfg, ScipyOpt),
    "torch": _OptimizerSpec(TorchOptCfg, TorchOpt),
    "mppi": _OptimizerSpec(MPPICfg, MPPI),
    "es": _OptimizerSpec(EvolutionStrategiesCfg, EvolutionStrategies),
}


def create_optimization_config(
    config_dict: Dict[str, Any],
    device_cfg: DeviceCfg,
) -> Any:
    """Create an optimizer configuration based on solver_type in the config dictionary.

    Args:
        config_dict: Dictionary containing optimizer configuration including solver_type.
        device_cfg: Tensor device type information.

    Returns:
        The appropriate optimizer configuration object.
    """
    solver_type = config_dict.get("solver_type", "")
    solver_spec = _SOLVER_REGISTRY.get(solver_type)
    if solver_spec is None:
        raise ValueError(
            f"Unsupported solver type: {solver_type}. Available: {list(_SOLVER_REGISTRY.keys())}"
        )
    config_dict = solver_spec.config_class.create_data_dict(config_dict.copy(), device_cfg)
    return solver_spec.config_class(**config_dict)


def create_optimizer(
    config: Any,
    rollout: List[Rollout],
    use_cuda_graph: bool = False,
) -> Any:
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

    solver_spec = _SOLVER_REGISTRY.get(solver_type)
    if solver_spec is None:
        raise ValueError(
            f"Unsupported solver type: {solver_type}. Available: {list(_SOLVER_REGISTRY.keys())}"
        )

    return solver_spec.optimizer_class(
        config=config,
        rollout_list=rollout,
        use_cuda_graph=use_cuda_graph,
    )
