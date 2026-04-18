# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unified collision buffer for sphere-obstacle distance computation.

This module provides a single buffer for all obstacle types (cuboid, mesh, voxel).
The generic collision kernels use atomic accumulation, allowing all obstacle types
to write to the same buffer without conflicts.
"""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import List, Optional, Union

# Third Party
import torch

# CuRobo
from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class CollisionBuffer:
    """Unified buffer for collision cost and gradient storage.

    This buffer stores the accumulated collision cost and gradients for all
    obstacle types. The generic collision kernels use atomic operations to
    accumulate results, so a single buffer is sufficient for all types.

    Tensor layouts:
        - distance: (batch, horizon, num_spheres) - accumulated collision cost
        - gradient: (batch, horizon, num_spheres, 4) - accumulated gradients [x, y, z, 0]
    """

    #: Accumulated collision distance/cost per sphere.
    distance: torch.Tensor

    #: Accumulated gradient per sphere [grad_x, grad_y, grad_z, 0].
    gradient: torch.Tensor

    #: Shape of the distance buffer.
    shape: Optional[torch.Size] = None

    #: Device configuration.
    device_cfg: DeviceCfg = None

    def __post_init__(self):
        """Initialize shape from distance buffer if not provided."""
        if self.shape is None:
            self.shape = self.distance.shape
        if self.device_cfg is None:
            self.device_cfg = DeviceCfg()

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_shape(
        cls,
        shape: torch.Size,
        device_cfg: DeviceCfg,
    ) -> CollisionBuffer:
        """Create a collision buffer from sphere query shape.

        Args:
            shape: Shape of query spheres (batch, horizon, num_spheres, 4).
            device_cfg: Device and dtype configuration.

        Returns:
            Initialized CollisionBuffer with zeroed tensors.
        """
        batch, horizon, num_spheres = shape[0], shape[1], shape[2]

        distance = torch.zeros(
            (batch, horizon, num_spheres),
            device=device_cfg.device,
            dtype=device_cfg.collision_distance_dtype,
        )
        gradient = torch.zeros(
            (batch, horizon, num_spheres, 4),
            device=device_cfg.device,
            dtype=device_cfg.collision_gradient_dtype,
        )

        return cls(
            distance=distance,
            gradient=gradient,
            shape=torch.Size([batch, horizon, num_spheres]),
            device_cfg=device_cfg,
        )

    # -------------------------------------------------------------------------
    # Buffer Management
    # -------------------------------------------------------------------------

    def zero_(self) -> None:
        """Zero all buffers in-place.

        Call this before launching collision kernels to reset accumulation.
        """
        self.distance.zero_()
        self.gradient.zero_()

    def resize(self, shape: Union[torch.Size, List[int]], device_cfg: DeviceCfg) -> None:
        """Resize buffers if shape has changed.

        Args:
            shape: New shape of query spheres (batch, horizon, num_spheres, 4).
            device_cfg: Device and dtype configuration.
        """
        new_shape = torch.Size([shape[0], shape[1], shape[2]])
        if self.shape != new_shape:
            self.distance = torch.zeros(
                new_shape,
                device=device_cfg.device,
                dtype=device_cfg.collision_distance_dtype,
            )
            self.gradient = torch.zeros(
                (*new_shape, 4),
                device=device_cfg.device,
                dtype=device_cfg.collision_gradient_dtype,
            )
            self.shape = new_shape
            self.device_cfg = device_cfg

    def clone(self) -> CollisionBuffer:
        """Create a deep copy of this buffer."""
        return CollisionBuffer(
            distance=self.distance.clone(),
            gradient=self.gradient.clone(),
            shape=self.shape,
            device_cfg=self.device_cfg,
        )

    # -------------------------------------------------------------------------
    # Arithmetic Operations
    # -------------------------------------------------------------------------

    def __mul__(self, scalar: float) -> CollisionBuffer:
        """Scale buffer values by a scalar."""
        self.distance *= scalar
        self.gradient *= scalar
        return self

    def __add__(self, other: CollisionBuffer) -> CollisionBuffer:
        """Add another buffer's values to this one."""
        self.distance += other.distance
        self.gradient += other.gradient
        return self

    # -------------------------------------------------------------------------
    # Accessors for Warp Kernels
    # -------------------------------------------------------------------------

    #def get_distance_flat(self) -> torch.Tensor:
    #    """Get flattened distance buffer for Warp kernels."""
    #    return self.distance.view(-1)

    #def get_gradient_flat(self) -> torch.Tensor:
    #    """Get flattened gradient buffer for Warp kernels."""
    #    return self.gradient.view(-1)
