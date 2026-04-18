# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""STOMP-specific particle processor for trajectory sampling."""

# Third Party
import torch

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise

# CuRobo
from .stomp_covariance import get_stomp_cov


class StompParticleProcessor:
    """STOMP-specific particle processor for smooth trajectory sampling.

    STOMP (Stochastic Trajectory Optimization for Motion Planning) uses
    a special covariance structure to generate smooth trajectory samples.
    This processor applies the STOMP transformation including:
    - Custom covariance matrix scaling
    - Trajectory reshaping and transposition
    - Boundary condition enforcement (zero velocity at start/end)
    - Normalization
    """

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        stencil_type: str = "3point",
        device_cfg: DeviceCfg = None,
    ):
        """Initialize STOMP particle processor.

        Args:
            horizon: Time horizon for trajectory sampling
            action_dim: Number of action dimensions (DOF)
            device_cfg: Device and dtype configuration
            stencil_type: Type of finite difference stencil to use
        """
        self.horizon = horizon
        self.action_dim = action_dim
        self.ndims = horizon * action_dim
        self.input_ndims = horizon * action_dim
        self.device_cfg = device_cfg
        self.stencil_type = stencil_type
        self.input_horizon = horizon

        # Get STOMP covariance matrices
        _, self.stomp_scale_tril, _ = get_stomp_cov(
            self.horizon,
            zero_out_boundary=True,
            stencil_type=stencil_type,
        )
        self.stomp_scale_tril = self.device_cfg.to_device(self.stomp_scale_tril)

    def process_samples(self, samples: torch.Tensor, filter_smooth: bool = False) -> torch.Tensor:
        """Process samples using STOMP transformation.

        Args:
            samples: Raw Gaussian samples with shape (batch, horizon , action_dim)
                    Note: Input should be flattened from generator output
            filter_smooth: Ignored for STOMP (inherently generates smooth trajectories)

        Returns:
            STOMP-processed trajectory samples with shape (batch, horizon, action_dim)
        """
        shape = samples.shape
        if min(shape) == 0:
            return samples
        batch_size = samples.shape[0]
        if len(samples.shape) != 3:
            log_and_raise("samples should be a 3D tensor")

        # Flatten samples to (batch, horizon * action_dim) if needed
        # if len(samples.shape) == 3:
        #    samples = samples.view(batch_size, self.horizon * self.action_dim)

        # Apply STOMP covariance transformation
        # stomp_scale_tril: (horizon, horizon)
        # samples: (batch, horizon * action_dim) -> (batch, horizon * action_dim, 1)
        transformed_samples = torch.matmul(self.stomp_scale_tril, samples)

        # Reshape and transpose to get proper trajectory format
        # (batch, horizon * action_dim) -> (batch, action_dim, horizon) -> (batch, horizon, action_dim)
        # trajectory_samples = transformed_samples.view(
        #    batch_size, self.action_dim, self.horizon
        # ).transpose(-2, -1)

        # Normalize by maximum absolute value across all dimensions
        max_abs = torch.max(torch.abs(transformed_samples))
        if max_abs > 0:
            transformed_samples = transformed_samples / max_abs

        # Enforce boundary conditions: zero velocity at start and end
        transformed_samples[:, 0, :] = 0.0  # First timestep: zero velocity
        transformed_samples[:, -2:, :] = 0.0  # Last two timesteps: zero velocity/acceleration

        # Check for NaN values which can indicate corrupted installation
        if torch.any(torch.isnan(transformed_samples)):
            log_and_raise("NaN values found in STOMP samples, installation could have been corrupted")

        return transformed_samples
