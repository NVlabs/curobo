#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimize the Rosenbrock function with L-BFGS and plot the optimization path.

This script creates a RosenbrockRollout instance and uses it with an optimizer
to find the minimum of the Rosenbrock function.

Output files are written to ``~/.cache/curobo/examples/custom_optimization/``
by default. Override with ``curobo.runtime.cache_dir``.
"""

from __future__ import annotations

# Standard Library
from pathlib import Path

# Third Party
import matplotlib.pyplot as plt
import numpy as np
import torch

# CuRobo
from curobo import runtime
from curobo.optim import LBFGSOpt, LBFGSOptCfg
from curobo.profiling import CudaEventTimer
from curobo.rollout import RosenbrockCfg, RosenbrockRollout
from curobo.types import DeviceCfg


def _get_output_dir() -> Path:
    """Return the example output directory, creating it if needed."""
    out = Path(runtime.cache_dir) / "examples" / "custom_optimization"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    # Set up device and tensor arguments
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)

    # Create Rosenbrock rollout configuration
    rosenbrock_config = RosenbrockCfg(
        a=1.0,
        b=100.0,
        dimensions=2,  # 2D Rosenbrock function
        time_horizon=1,  # Single time step for this example
        time_action_horizon=1,
        device_cfg=device_cfg,
    )

    # L-BFGS configuration.
    opt_config_dict = {
        "num_iters": 50,
        "line_search_scale": [0.01, 0.2, 0.3, 0.5, 0.75, 0.8, 0.9, 1.0, 2.0],
        "cost_convergence": 0.0,
        "cost_delta_threshold": 0.0,
        "fixed_iters": True,
        "store_debug": True,
        "history": 2,
        "epsilon": 0.001,
        "sync_cuda_time": True,
        "debug_info": None,
        "num_problems": 1,
        "use_coo_sparse": True,
        "solver_type": "lbfgs",
        "inner_iters": 1,
        "line_search_type": "wolfe",
        "line_search_wolfe_c_1": 0.001,
        "line_search_wolfe_c_2": 0.98,
        "step_scale": 1.0,
        "cost_relative_threshold": 0.999,
    }
    opt_config = LBFGSOptCfg(**LBFGSOptCfg.create_data_dict(opt_config_dict, device_cfg))
    lbfgs_rollouts = [
        RosenbrockRollout(rosenbrock_config, use_cuda_graph=True)
        for _ in range(opt_config.num_rollout_instances)
    ]
    optimizer = LBFGSOpt(opt_config, lbfgs_rollouts, use_cuda_graph=True)
    rosenbrock_rollout = lbfgs_rollouts[0]

    batch_size = 1
    # update_num_problems resizes the optimizer's internal rollouts (e.g. to
    # ``len(line_search_scale)`` for L-BFGS), so set ``rosenbrock_rollout.batch_size``
    # afterwards to control the shape of ``get_initial_action`` below.
    optimizer.update_num_problems(batch_size)
    rosenbrock_rollout.batch_size = batch_size
    # Get initial action sequence
    init_action = rosenbrock_rollout.get_initial_action()
    print("Starting optimization...")

    # Run optimization twice so the second run benefits from the warmed-up CUDA
    # graphs / solver state captured on the first run.
    for _ in range(2):
        optimizer.reinitialize(init_action)

        timer = CudaEventTimer().start()
        optimized_action = optimizer.optimize(init_action)
        elapsed = timer.stop()
    print(f"Optimization completed in {elapsed} seconds")

    optimized_cost = rosenbrock_rollout.evaluate_action(
        optimized_action
    ).costs_and_constraints.get_sum_cost()

    # Print results
    print(f"Optimized cost: {optimized_cost.cpu().numpy()}")
    print(f"Optimized action: {optimized_action.cpu().numpy()}")
    print("True minimum: [1.0, 1.0]")

    # Visualize the Rosenbrock function and optimization path
    action_history = optimizer.get_recorded_trace()["debug"]

    visualize_batch_rosenbrock(action_history, rosenbrock_config, plot_path=True)


def visualize_batch_rosenbrock(
    action_history: list[torch.Tensor], config: RosenbrockCfg, plot_path: bool = True
) -> None:
    """Visualize the Rosenbrock function and optimization path.

    Args:
        action_history: List of action tensors with shape (batch_size, horizon, action_dim)
        config: RosenbrockCfg object containing function parameters
    """
    # Convert action history to numpy array
    action_history = torch.stack([x for x in action_history])
    path = action_history.cpu().numpy()

    # Create a grid of points for visualization
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    # Compute Rosenbrock function values
    Z = (config.a - X) ** 2 + config.b * (Y - X**2) ** 2

    # Get batch size and number of iterations
    batch_size = path.shape[1]

    # Create plot
    plt.figure(figsize=(8, 5))

    # Plot contour of Rosenbrock function
    contour = plt.contourf(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap="viridis", alpha=0.4)
    plt.contour(
        X,
        Y,
        Z,
        levels=np.logspace(-1, 3, 20),
        colors="black",
        alpha=0.1,
    )

    plt.colorbar(contour, label="Function Value")

    # Plot optimization paths for each batch element with different colors
    colors = plt.cm.inferno(np.linspace(0, 1, batch_size))
    if plot_path:
        for i in range(batch_size):
            # Extract path for this batch element (iterations, horizon, action_dim)
            batch_path = path[:, i, 0, :]  # Take first horizon step

            # Plot optimization path
            plt.plot(
                batch_path[:, 0],
                batch_path[:, 1],
                ".-",
                color=colors[i],
                linewidth=1.0,
                markersize=5,
                label=f"Seed {i + 1}",
            )

            # Mark start and end points
            plt.plot(batch_path[0, 0], batch_path[0, 1], "o", color=colors[i], markersize=8)
            plt.plot(batch_path[-1, 0], batch_path[-1, 1], "s", color=colors[i], markersize=10)

    # Mark true minimum
    plt.plot(1, 1, "r*", markersize=20, label="True Minimum")

    # Add labels and legend
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Rosenbrock Function")
    plt.legend()
    plt.grid(True)
    plt.xlim(-2, 2.0)
    plt.ylim(-1.5, 2.0)
    plt.tight_layout()
    out = _get_output_dir()
    if plot_path:
        save_path = out / "rosenbrock_optimization.png"
    else:
        save_path = out / "rosenbrock_function.png"
    plt.savefig(str(save_path))
    print(f"Plot saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
