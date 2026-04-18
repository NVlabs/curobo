# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Standard particle processor for filtering samples."""

from typing import Optional, Tuple

# Third Party
import torch

from curobo._src.types.device_cfg import DeviceCfg

# CuRobo
from .stomp_covariance import get_stomp_cov


class StandardParticleProcessor:
    """Standard filtering particle processor used by Halton and Random particle samplers.

    This processor applies standard filtering techniques including:
    - Filter coefficients for temporal smoothing
    - STOMP matrix filtering for smooth trajectories

    Both filtering methods are inherited from the original BaseSampleLib implementation.
    """

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        filter_coeffs: Optional[Tuple[float, float, float]] = None,
        device_cfg: DeviceCfg = None,
    ):
        """Initialize standard particle processor.

        Args:
            horizon: Time horizon for trajectory sampling
            action_dim: Number of action dimensions (DOF)
            device_cfg: Device and dtype configuration
            filter_coeffs: Optional filter coefficients (beta_0, beta_1, beta_2) for temporal filtering
        """
        self.horizon = horizon
        self.action_dim = action_dim
        self.device_cfg = device_cfg
        self.filter_coeffs = filter_coeffs
        self.ndims = horizon * action_dim
        self.input_ndims = horizon * action_dim
        self.input_horizon = horizon
        # Lazy initialization for STOMP matrices
        self.stomp_matrix = None
        self.stomp_scale_tril = None

    def process_samples(self, samples: torch.Tensor, filter_smooth: bool = False) -> torch.Tensor:
        """Process samples using either standard filtering or smooth STOMP filtering.

        Args:
            samples: Raw samples from generator with shape (batch, horizon, action_dim)
            filter_smooth: Whether to use STOMP smooth filtering (True) or standard filtering (False)

        Returns:
            Processed samples with same shape as input
        """
        if filter_smooth:
            return self._filter_smooth(samples)
        else:
            return self._filter_samples(samples)

    def _filter_samples(self, eps: torch.Tensor) -> torch.Tensor:
        """Apply standard temporal filtering using filter coefficients.

        Args:
            eps: Input samples with shape (batch, horizon, action_dim)

        Returns:
            Filtered samples with same shape
        """
        if self.filter_coeffs is not None:
            beta_0, beta_1, beta_2 = self.filter_coeffs

            # Apply temporal filtering (could be tensorized for better performance)
            for i in range(2, eps.shape[1]):
                eps[:, i, :] = (
                    beta_0 * eps[:, i, :] + beta_1 * eps[:, i - 1, :] + beta_2 * eps[:, i - 2, :]
                )
        return eps

    def _filter_smooth(self, samples: torch.Tensor) -> torch.Tensor:
        """Apply STOMP matrix smooth filtering.

        Args:
            samples: Input samples with shape (batch, horizon, action_dim)

        Returns:
            Smoothly filtered samples with same shape
        """
        # Handle empty samples
        if samples.shape[0] == 0:
            return samples

        # Lazy initialization of STOMP matrices
        if self.stomp_matrix is None:
            self.stomp_matrix, self.stomp_scale_tril, _ = get_stomp_cov(
                self.horizon, zero_out_boundary=True, stencil_type="3point"
            )
            self.stomp_matrix = self.device_cfg.to_device(self.stomp_matrix)
            self.stomp_scale_tril = self.device_cfg.to_device(self.stomp_scale_tril)

        # Apply STOMP matrix filtering
        # stomp_matrix: (horizon, horizon)
        # samples: (batch, horizon, action_dim)
        filter_samples = torch.matmul(self.stomp_matrix, samples)
        # Normalize by maximum absolute value
        filter_samples = filter_samples / torch.max(torch.abs(filter_samples))

        return filter_samples
