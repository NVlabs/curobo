# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Configuration dataclass for SDFPoseDetector."""

from dataclasses import dataclass, field

from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class SDFDetectorCfg:
    """Configuration for SDF-based pose detector (Levenberg-Marquardt optimization).

    Uses mesh SDF queries for implicit correspondence and analytic gradients.
    """

    # Optimization parameters
    max_iterations: int = 100
    inner_iterations: int = 25  # Iterations per CUDA graph capture (no convergence check)
    convergence_threshold: float = 1e-5  # Translation convergence (meters)
    rotation_convergence_threshold: float = 1e-5  # Rotation convergence (radians)

    # CUDA graph acceleration
    use_cuda_graph: bool = True  # Enable CUDA graph for inner iterations

    # Correspondence parameters
    #max_distance: float = 0.2  # Maximum SDF query distance (meters)
    distance_threshold: float = 0.2  # Reject correspondences beyond this (meters)
    min_valid_ratio: float = 0.1  # Minimum ratio of valid correspondences

    # Robust estimation
    use_huber: bool = True  # Use Huber loss for outlier robustness
    huber_delta: float = 0.1  # Huber threshold in meters (20mm)

    # Levenberg-Marquardt parameters
    lambda_initial: float = 1e-3  # Initial damping parameter
    lambda_factor: float = 10.0  # Factor to multiply/divide lambda
    lambda_min: float = 1e-7  # Minimum lambda value
    lambda_max: float = 1e7  # Maximum lambda value
    rho_min: float = 0.25  # Minimum trust ratio for step acceptance

    # Point sampling
    n_points: int = 5000  # Number of observed points to use

    device_cfg: DeviceCfg = field(default_factory=DeviceCfg)

    @property
    def max_distance(self):
        return self.distance_threshold
