#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Example script demonstrating the use of RosenbrockRollout for optimization.

This script creates a RosenbrockRollout instance and uses it with an optimizer
to find the minimum of the Rosenbrock function.

Output files are written to ``~/.cache/curobo/examples/custom_optimization/``
by default. Override with ``curobo.runtime.cache_dir``.
"""

# Standard Library
from pathlib import Path

# Third Party
import matplotlib.pyplot as plt
import numpy as np
import torch

# CuRobo
from curobo import runtime
from curobo.optim import (
    MPPI,
    EvolutionStrategies,
    EvolutionStrategiesCfg,
    LBFGSOpt,
    LBFGSOptCfg,
    MPPICfg,
    MultiStageOptimizer,
    ScipyOpt,
    ScipyOptCfg,
    TorchOpt,
    TorchOptCfg,
)
from curobo.profiling import CudaEventTimer
from curobo.rollout import RosenbrockCfg, RosenbrockRollout
from curobo.types import DeviceCfg


def _get_output_dir() -> Path:
    """Return the example output directory, creating it if needed."""
    out = Path(runtime.cache_dir) / "examples" / "custom_optimization"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_rollout_list(rosenbrock_config, num_instances):
    """Return ``num_instances`` Rosenbrock rollouts sharing the same config."""
    return [
        RosenbrockRollout(rosenbrock_config, use_cuda_graph=True)
        for _ in range(num_instances)
    ]


def load_torch_opt(rosenbrock_config, device_cfg):
    opt_config_dict = {
        "num_iters": 200,
        "solver_type": "torch",
        "store_debug": True,
        "torch_optim_name": "Adam",
        "torch_optim_kwargs": {"lr": 0.01, "weight_decay": 0.0},
    }
    opt_config_dict = TorchOptCfg.create_data_dict(opt_config_dict, device_cfg)
    opt_config = TorchOptCfg(**opt_config_dict)
    rollout_list = _build_rollout_list(rosenbrock_config, opt_config.num_rollout_instances)
    return TorchOpt(opt_config, rollout_list), rollout_list


def load_scipy_opt(rosenbrock_config, device_cfg):
    opt_config_dict = {
        "num_iters": 100,
        "solver_name": "scipy",
        "solver_type": "scipy",
        "store_debug": True,
        "scipy_minimize_method": "L-BFGS-B",
    }
    opt_config_dict = ScipyOptCfg.create_data_dict(opt_config_dict, device_cfg)
    opt_config = ScipyOptCfg(**opt_config_dict)
    rollout_list = _build_rollout_list(rosenbrock_config, opt_config.num_rollout_instances)
    return ScipyOpt(opt_config, rollout_list), rollout_list


def main():
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

    # Build per-optimizer rollout lists. Each optimizer owns a list of
    # ``num_rollout_instances`` rollouts that all share ``rosenbrock_config``;
    # ``use_cuda_graph=True`` records the rollout once and replays it on each
    # evaluation for lower kernel-launch overhead.
    mppi_config_dict = {
        "num_iters": 10,
        "inner_iters": 5,
        "gamma": 1.0,
        "seed": 0,
        "store_rollouts": False,
        "num_particles": 1000,
        "solver_type": "mppi",
        "store_debug": True,
        "sample_mode": "BEST",
        "init_cov": 0.01,
        "kappa": 0.0001,
        "beta": 0.1,
        "step_size_mean": 0.9,
        "step_size_cov": 0.1,
        "null_act_frac": 0.0,
        "squash_fn": "CLAMP",
        "cov_type": "DIAG_A",
    }
    mppi_config = MPPICfg(**MPPICfg.create_data_dict(mppi_config_dict, device_cfg))
    mppi_rollouts = _build_rollout_list(rosenbrock_config, mppi_config.num_rollout_instances)
    mppi = MPPI(mppi_config, mppi_rollouts)

    es_config_dict = mppi_config_dict.copy()
    es_config_dict["learning_rate"] = 0.25
    es_config_dict["update_cov"] = False
    es_config_dict["init_cov"] = 0.25
    es_config_dict["sample_mode"] = "BEST"
    es_config = EvolutionStrategiesCfg(
        **EvolutionStrategiesCfg.create_data_dict(es_config_dict, device_cfg)
    )
    es_rollouts = _build_rollout_list(rosenbrock_config, es_config.num_rollout_instances)
    evolution_strategies = EvolutionStrategies(es_config, es_rollouts)

    # L-BFGS configuration.
    opt_config_dict = {
        "num_iters": 50,
        "line_search_scale": [0.0, 0.01, 0.2, 0.3, 0.5, 0.75, 0.8, 0.9, 1.0, 2.0],
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
    lbfgs_rollouts = _build_rollout_list(rosenbrock_config, opt_config.num_rollout_instances)
    lbfgs_opt = LBFGSOpt(opt_config, lbfgs_rollouts, use_cuda_graph=False)

    # Select which optimizer to run. Swap the assignment below to try MPPI /
    # Evolution Strategies / L-BFGS / Torch instead of SciPy.
    # optimizer, rollout_list = mppi, mppi_rollouts
    # optimizer, rollout_list = evolution_strategies, es_rollouts
    optimizer, rollout_list = lbfgs_opt, lbfgs_rollouts
    # optimizer, rollout_list = load_torch_opt(rosenbrock_config, device_cfg)
    # optimizer, rollout_list = load_scipy_opt(rosenbrock_config, device_cfg)

    rosenbrock_rollout = rollout_list[0]
    multi_stage_optimizer = MultiStageOptimizer(optimizers=[optimizer])
    optimizer = multi_stage_optimizer

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
        result = optimizer.optimize(init_action)
        elapsed = timer.stop()
    print(f"Optimization completed in {elapsed} seconds")

    # Extract optimized solution
    optimized_action = result
    print(optimized_action.shape)
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


def visualize_batch_rosenbrock_3d(action_history, config, plot_path=True):
    """Visualize the Rosenbrock function and optimization path in 3D.

    Args:
        action_history: List of action tensors with shape (batch_size, horizon, action_dim)
        config: RosenbrockCfg object containing function parameters
    """
    # Third Party

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

    # Create 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the Rosenbrock function surface
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="viridis",
        alpha=0.2,
        linewidth=1,
        antialiased=False,
        rcount=100,
        ccount=100,
        vmax=1000,
    )

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="Function Value")

    # Plot optimization paths for each batch element with different colors
    colors = plt.cm.inferno(np.linspace(0, 1, batch_size))

    if plot_path:
        for i in range(batch_size):
            # Extract path for this batch element (iterations, horizon, action_dim)
            batch_path = path[:, i, 0, :]  # Take first horizon step

            # Get z-values along the path
            z_values = np.array(
                [
                    (config.a - x) ** 2 + config.b * (y - x**2) ** 2
                    for x, y in zip(batch_path[:, 0], batch_path[:, 1])
                ]
            )

            # Plot optimization path
            ax.plot(
                batch_path[:, 0],
                batch_path[:, 1],
                z_values,
                ".-",
                color=colors[i],
                linewidth=2.0,
                markersize=5,
                label=f"Seed {i + 1}",
            )

            # Mark start and end points
            ax.scatter(
                batch_path[0, 0], batch_path[0, 1], z_values[0], color=colors[i], s=80, marker="o"
            )
            ax.scatter(
                batch_path[-1, 0],
                batch_path[-1, 1],
                z_values[-1],
                color=colors[i],
                s=80,
                marker="s",
            )

    # Mark true minimum
    min_z = (config.a - 1) ** 2 + config.b * (1 - 1**2) ** 2  # Should be 0
    ax.scatter(1, 1, min_z, color="r", s=1000, marker="*", label="True Minimum")

    # Add labels and legend
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    ax.set_title("Rosenbrock Function - 3D Visualization")
    ax.legend()

    # Set view angle
    ax.view_init(elev=20, azim=75)

    # Set axis limits
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 2)
    ax.set_zlim(0, 1000)  # Adjust as needed to show the relevant part of the function
    plt.tight_layout()
    # plt.show()
    # Save the figure
    save_path = _get_output_dir() / "rosenbrock_optimization_3d.png"
    plt.savefig(str(save_path))
    print(f"3D visualization saved to: {save_path}")
    plt.close()


# Call both visualization functions


def visualize_batch_rosenbrock(action_history, config, plot_path=True):
    """Visualize the Rosenbrock function and optimization path.

    Args:
        action_history: List of action tensors with shape (batch_size, horizon, action_dim)
        config: RosenbrockCfg object containing function parameters
    """
    # Third Party

    # Convert action history to numpy array
    action_history = torch.stack([x for x in action_history])
    path = action_history.cpu().numpy()
    print(path.shape)

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
        # cmap='viridis_r',
        alpha=0.1,
    )

    plt.colorbar(contour, label="Function Value")

    # Plot optimization paths for each batch element with different colors
    # Use a more uniform colormap for better visualization
    # Options: viridis, plasma, inferno, magma, cividis
    colors = plt.cm.inferno(np.linspace(0, 1, batch_size))
    # colors = plt.cm.tab10(np.linspace(0, 1, batch_size))
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
    # plt.show()
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
