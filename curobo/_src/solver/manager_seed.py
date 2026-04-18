# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Action seed manager for optimization-based solvers."""

from __future__ import annotations

# Standard Library
from typing import Optional

# Third Party
import torch

# CuRobo
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise, log_warn
from curobo._src.util.sampling import SampleBuffer
from curobo._src.util.trajectory_seed_generator import TrajectorySeedGenerator


class SeedManager:
    """Handles generation and preparation of action seeds for optimization.

    This class manages seed generation for both single-step (IK) and multi-step
    (trajectory optimization) problems.
    """

    def __init__(
        self,
        device_cfg: DeviceCfg,
        action_dim: int,
        action_bound_lows: torch.Tensor,
        action_bound_highs: torch.Tensor,
        random_seed: int = 123,
        action_horizon: int = 1,
    ):
        """Initialize the seed manager.

        Args:
            device_cfg: Device and dtype configuration.
            action_dim: Dimensions of the action space (e.g., DOF).
            action_bound_lows: Lower bounds for action space.
            action_bound_highs: Upper bounds for action space.
            random_seed: Seed for random number generation.
            action_horizon: Horizon for trajectory seeds (1 for single actions).
        """
        self.device_cfg = device_cfg
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # Initialize Halton sequence generator for random actions
        self.action_sample_generator = SampleBuffer.create_roberts_sample_buffer(
            ndims=action_dim,
            device_cfg=device_cfg,
            up_bounds=action_bound_highs,
            low_bounds=action_bound_lows,
        )
        self.action_sample_generator = SampleBuffer.create_halton_sample_buffer(
            ndims=action_dim,
            device_cfg=device_cfg,
            up_bounds=action_bound_highs,
            low_bounds=action_bound_lows,
            seed=random_seed,
            store_buffer=2000,
        )

        # For trajectory seeds, initialize trajectory seed generator if needed
        self.trajectory_seed_generator = None
        if action_horizon > 1:
            self.trajectory_seed_generator = TrajectorySeedGenerator(
                action_horizon, action_dim, device_cfg
            )

    def prepare_action_seeds(
        self,
        batch_size: int,
        num_seeds: int,
        seed_config: Optional[torch.Tensor] = None,
        current_state: Optional[JointState] = None,
        seed_traj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Prepare single-step action seeds (joint configurations) for optimization.

        When provided seed_config are less than num_seeds, remaining seeds are
        generated randomly. Handles validation and reshaping of seeds.

        Args:
            batch_size: Number of parallel optimization problems.
            num_seeds: Total number of seeds required per batch item.
            seed_config: Optional user-provided seeds. Shape (batch, n, dof) or (n, batch, dof).
            current_state: Optional current state (used in some contexts).
            seed_traj: Optional seed trajectories (used in some contexts).

        Returns:
            Seed configurations tensor of shape (batch * num_seeds, 1, dof).
        """
        if current_state is not None or seed_traj is not None:
            log_warn(
                "SeedManager.prepare_action_seeds ignores current_state and seed_traj arguments."
            )

        # Ensure seed_config has shape (batch_size, n, dof) if provided
        if seed_config is not None:
            # Support both (n, batch, dof) and (batch, n, dof) formats
            if seed_config.shape[0] != batch_size and seed_config.shape[1] == batch_size:
                seed_config = seed_config.permute(1, 0, 2)  # Transpose to (batch, n, dof)
            elif seed_config.shape[0] != batch_size:
                log_and_raise(
                    f"seed_config batch dimension ({seed_config.shape[0]}) "
                    f"does not match expected batch size ({batch_size})"
                )

            n_provided_seeds = seed_config.shape[1]
            if n_provided_seeds > num_seeds:
                log_warn(
                    f"Provided {n_provided_seeds} seeds, but only {num_seeds} are required. "
                    f"Using the first {num_seeds}."
                )
                seed_config = seed_config[:, :num_seeds, :]
                n_provided_seeds = num_seeds

            if n_provided_seeds == num_seeds:
                action_seeds = seed_config
            else:
                n_random_seeds = num_seeds - n_provided_seeds
                random_seeds = self.generate_random_actions(batch_size, n_random_seeds)
                action_seeds = torch.cat((seed_config, random_seeds), dim=1)
        else:
            action_seeds = self.generate_random_actions(batch_size, num_seeds)

        # Reshape for optimizer input: (batch * num_seeds, 1, dof)
        action_seeds = action_seeds.reshape(-1, 1, self.action_dim)
        return action_seeds

    def prepare_trajectory_seeds(
        self,
        batch_size: int,
        num_seeds: int,
        current_state: JointState,
        seed_config: Optional[torch.Tensor] = None,
        seed_traj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Prepare multi-step action seeds (trajectories) for optimization.

        Seeds can come from:
        1. Provided seed trajectories (seed_traj)
        2. Interpolation between current_state and seed_config
        3. Constant trajectory at current position if no seed_config is provided.

        Args:
            batch_size: Number of parallel optimization problems.
            num_seeds: Number of seeds needed per batch.
            current_state: Current joint state (batch_size, dof).
            seed_config: Optional target joint configs (batch_size, n, dof).
            seed_traj: Optional full seed trajectories (batch_size, n, horizon, dof).

        Returns:
            Seed trajectories tensor (batch_size * num_seeds, horizon, dof).
        """
        if self.trajectory_seed_generator is None:
            log_and_raise(
                "Cannot prepare trajectory seeds: trajectory_seed_generator not initialized"
            )

        if current_state.shape[0] != batch_size:
            log_and_raise(
                f"Current state batch ({current_state.shape[0]}) != batch_size ({batch_size})"
            )

        generated_seeds = []
        seeds_remaining = num_seeds

        # 1. Use provided seed trajectories if available
        if seed_traj is not None:
            expected_shape = f"({batch_size}, n, {self.action_horizon}, {self.action_dim})"
            if (
                seed_traj.ndim != 4
                or seed_traj.shape[0] != batch_size
                or seed_traj.shape[2] != self.action_horizon
                or seed_traj.shape[3] != self.action_dim
            ):
                log_and_raise(
                    f"Invalid seed_traj shape {seed_traj.shape}. Expected {expected_shape}"
                )

            n_available = seed_traj.shape[1]
            n_to_use = min(n_available, seeds_remaining)
            generated_seeds.append(seed_traj[:, :n_to_use])
            seeds_remaining -= n_to_use

        # 2. Generate additional seeds if needed
        if seeds_remaining > 0:
            if seed_config is not None:
                n_configs = seed_config.shape[1]
                if n_configs < seeds_remaining:
                    log_and_raise(
                        f"Insufficient seed configs: {n_configs} provided, {seeds_remaining} needed"
                    )

                configs_to_use = seed_config[:, :seeds_remaining]
                interpolated_seeds = self.trajectory_seed_generator.generate_interpolated_seeds(
                    current_state.position, configs_to_use, seeds_remaining
                )
                generated_seeds.append(interpolated_seeds)
            else:
                constant_seeds = self.trajectory_seed_generator.generate_constant_seeds(
                    current_state.position, seeds_remaining
                )
                generated_seeds.append(constant_seeds)

        all_seeds = torch.cat(generated_seeds, dim=1)
        return all_seeds.view(-1, self.action_horizon, self.action_dim)

    def generate_random_actions(
        self,
        batch_size: int,
        num_seeds: int,
    ) -> torch.Tensor:
        """Generate random action seeds using the Halton generator.

        Args:
            batch_size: Number of batch items.
            num_seeds: Number of seeds to generate per batch item.

        Returns:
            Tensor of seed configurations with shape (batch_size, num_seeds, dof).
        """
        if num_seeds <= 0:
            return torch.zeros(
                (batch_size, 1, self.action_dim),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )

        random_seeds = self.action_sample_generator.get_samples(
            num_seeds * batch_size,
            bounded=True,
        ).view(batch_size, num_seeds, self.action_dim)

        return random_seeds

    def prepare_deceleration_trajectory_seeds(
        self,
        batch_size: int,
        num_seeds: int,
        current_state: JointState,
        deceleration_time: Optional[float] = None,
        deceleration_profile: str = "exponential",
    ) -> torch.Tensor:
        """Prepare deceleration trajectory seeds that gradually bring robot to stop.

        Args:
            batch_size: Number of parallel optimization problems.
            num_seeds: Number of seeds needed per batch.
            current_state: Current joint state with velocity information.
            deceleration_time: Time to decelerate (uses full horizon if None).
            deceleration_profile: Type of deceleration curve.

        Returns:
            Deceleration seed trajectories (batch_size * num_seeds, horizon, dof).
        """
        if self.trajectory_seed_generator is None:
            log_and_raise(
                "Cannot prepare trajectory seeds: trajectory_seed_generator not initialized"
            )

        if current_state.shape[0] != batch_size:
            log_and_raise(
                f"Current state batch ({current_state.shape[0]}) != batch_size ({batch_size})"
            )

        deceleration_seeds = self.trajectory_seed_generator.generate_deceleration_seeds(
            current_state,
            num_seeds,
            deceleration_time=deceleration_time,
            deceleration_profile=deceleration_profile,
        )

        return deceleration_seeds.view(-1, self.action_horizon, self.action_dim)

    def reset_seed(self) -> None:
        """Reset the random seed generators."""
        self.action_sample_generator.reset()

