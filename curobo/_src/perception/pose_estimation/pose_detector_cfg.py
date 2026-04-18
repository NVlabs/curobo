# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Configuration dataclass for PoseDetector."""

from dataclasses import dataclass, field

from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class DetectorCfg:
    """Configuration for pose detector (point-to-plane ICP with Huber loss)."""

    # Coarse stage parameters
    n_mesh_points_coarse: int = 500
    n_observed_points_coarse: int = 2000
    n_rotation_samples: int = 64
    n_iterations_coarse: int = 50
    distance_threshold_coarse: float = 0.5

    # Fine stage parameters
    n_mesh_points_fine: int = 2000
    n_observed_points_fine: int = 10000
    n_iterations_fine: int = 50
    distance_threshold_fine: float = 0.01

    # Solver method: Cholesky (default, faster) or SVD (more accurate)
    use_svd: bool = False

    # Robust estimation
    use_huber_loss: bool = True  # Use Huber loss for outlier robustness
    huber_delta: float = 0.02  # Huber threshold in meters (20mm)

    # Debug options
    save_iterations: bool = False  # Save iteration history for visualization

    device_cfg: DeviceCfg = field(default_factory=DeviceCfg)
