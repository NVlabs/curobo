# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Knot-based B-spline particle processor for trajectory sampling."""

# Third Party
import numpy as np
import scipy.interpolate as si
import torch

# CuRobo
from curobo._src.types.device_cfg import DeviceCfg


class KnotParticleProcessor:
    """B-spline knot interpolation particle processor.

    This processor treats generated samples as knot points and interpolates
    smooth B-spline trajectories from them. This is useful for generating
    smooth, continuous trajectories for robotic motion planning.

    The process:
    1. Generated samples are treated as knot points in control space
    2. B-spline interpolation creates smooth trajectories
    3. Output trajectories have the specified horizon length
    """

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        n_knots: int,
        degree: int = 3,
        device_cfg: DeviceCfg = None,
    ):
        """Initialize knot particle processor.

        Args:
            horizon: Time horizon for output trajectories
            action_dim: Number of action dimensions (DOF)
            n_knots: Number of knot points for B-spline interpolation
            degree: B-spline degree for interpolation
            device_cfg: Device and dtype configuration
        """
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_knots = n_knots
        self.degree = degree
        self.ndims = horizon * action_dim
        self.input_ndims = n_knots * action_dim
        self.input_horizon = n_knots
        self.device_cfg = device_cfg or DeviceCfg()

    def process_samples(self, samples: torch.Tensor, filter_smooth: bool = False) -> torch.Tensor:
        """Process knot samples into smooth B-spline trajectories.

        Args:
            samples: Raw knot point samples with shape (batch, n_knots * action_dim)
                    Note: Input should be reshaped from generator output for knots
            filter_smooth: Ignored for knot processing (B-splines are inherently smooth)

        Returns:
            Smooth trajectory samples with shape (batch, horizon, action_dim)
        """
        batch_size = samples.shape[0]

        # Reshape samples to (batch, action_dim, n_knots) for knot interpolation
        knot_samples = samples.view(batch_size, self.action_dim, self.n_knots)

        # Initialize output trajectory samples
        trajectory_samples = torch.zeros(
            (batch_size, self.horizon, self.action_dim),
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )

        # Interpolate B-splines for each sample and action dimension
        for i in range(batch_size):
            for j in range(self.action_dim):
                trajectory_samples[i, :, j] = self.bspline(
                    knot_samples[i, j, :], n=self.horizon, degree=self.degree
                )

        return trajectory_samples

    @staticmethod
    def bspline(c_arr: torch.Tensor, t_arr=None, n=100, degree=3):
        """Generate B-spline interpolation from control points.

        Args:
            c_arr: Control points tensor
            t_arr: Optional time array for parameterization
            n: Number of output points
            degree: B-spline degree

        Returns:
            Interpolated B-spline curve as tensor
        """
        sample_device = c_arr.device
        sample_dtype = c_arr.dtype
        cv = c_arr.cpu().numpy()

        if t_arr is None:
            t_arr = np.linspace(0, cv.shape[0], cv.shape[0])
        else:
            t_arr = t_arr.cpu().numpy()

        # Ensure degree is valid for the number of data points
        # For scipy B-spline, we need at least k+1 data points for degree k
        max_degree = max(1, cv.shape[0] - 1)
        actual_degree = min(degree, max_degree)

        spl = si.splrep(t_arr, cv, k=actual_degree, s=0.5)

        xx = np.linspace(0, cv.shape[0], n)
        samples = si.splev(xx, spl, ext=3)
        samples = torch.as_tensor(samples, device=sample_device, dtype=sample_dtype)

        return samples
