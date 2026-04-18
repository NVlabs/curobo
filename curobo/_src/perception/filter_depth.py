# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Depth image filtering for removing sensor noise and artifacts.

This module provides GPU-accelerated depth filtering using fused Warp kernels.
Filters are applied in a single GPU pass for efficiency.

Filters included:
    - Range filtering: Clip depth values to valid range
    - Flying pixel detection: Remove artifacts at depth discontinuities
    - Bilateral filtering: Smooth depth while preserving edges

The kernels (and :class:`FilterDepth`) operate strictly on batched
``(B, H, W)`` tensors. Single-image callers should explicitly promote with
``depth.unsqueeze(0)`` and index ``filtered[0]`` on return.

Example:
    >>> from curobo._src.perception.filter_depth import FilterDepth
    >>> # Single image (B=1)
    >>> depth_filter = FilterDepth(image_shape=(480, 640))
    >>> filtered, valid = depth_filter(depth_image.unsqueeze(0))      # (1, H, W)
    >>> filtered, valid = filtered[0], valid[0]                       # (H, W)
    >>> # Batch of 2 cameras (single kernel launch)
    >>> depth_filter = FilterDepth(image_shape=(480, 640), num_batch=2)
    >>> filtered, valid = depth_filter(depth_batch)                   # (B, H, W)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from curobo._src.curobolib.cuda_ops.tensor_checks import check_float32_tensors
from curobo._src.util.logging import log_and_raise


@dataclass
class FilterDepthConfig:
    """Configuration for depth filtering.

    Attributes:
        depth_minimum_distance: Minimum valid depth in meters. Default: 0.1m.
        depth_maximum_distance: Maximum valid depth in meters. Default: 10.0m.
        flying_pixel_threshold: Filter aggressiveness 0.0-1.0 for flying pixels.
            Higher = more aggressive. Set to None to disable. Default: 0.5.
        bilateral_kernel_size: Size of bilateral kernel (must be odd).
            Set to None to disable bilateral filtering. Default: 5.
        bilateral_sigma_spatial: Spatial sigma in pixels. Default: 2.0.
        bilateral_sigma_depth: Depth sigma in meters. Default: 0.05.
    """

    depth_minimum_distance: float = 0.1
    depth_maximum_distance: float = 10.0
    flying_pixel_threshold: Optional[float] = 0.5
    bilateral_kernel_size: Optional[int] = 5
    bilateral_sigma_spatial: float = 2.0
    bilateral_sigma_depth: float = 0.05


class FilterDepth:
    """GPU-accelerated depth filtering using fused Warp kernels.

    Combines range filtering, flying pixel detection, and bilateral smoothing
    into a single efficient GPU pass. Pre-allocates buffers at initialization
    for zero-allocation filtering during runtime.

    Operates strictly on batched depth tensors of shape ``(B, H, W)``. Set
    ``num_batch > 1`` at init to pre-allocate batched output buffers so all
    images in the batch are filtered in a single kernel launch. Single-image
    callers must explicitly pass ``depth.unsqueeze(0)`` and index
    ``filtered[0]`` on return.

    Args:
        image_shape: Tuple of ``(height, width)`` for each depth image.
        depth_minimum_distance: Minimum valid depth in meters. Default: 0.1m.
        depth_maximum_distance: Maximum valid depth in meters. Default: 10.0m.
        flying_pixel_threshold: Filter aggressiveness 0.0-1.0 for flying pixels.
            Higher = more aggressive. Set to None to disable. Default: 0.5.
        bilateral_kernel_size: Size of bilateral kernel (must be odd).
            Set to None to disable bilateral filtering. Default: 5.
        bilateral_sigma_spatial: Spatial sigma in pixels. Default: 2.0.
        bilateral_sigma_depth: Depth sigma in meters. Default: 0.05.
        device: CUDA device. Default: "cuda".
        num_batch: Pre-allocate buffers for this batch size. Default: 1.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int],
        depth_minimum_distance: float = 0.1,
        depth_maximum_distance: float = 10.0,
        flying_pixel_threshold: Optional[float] = 0.5,
        bilateral_kernel_size: Optional[int] = 5,
        bilateral_sigma_spatial: float = 10.0,
        bilateral_sigma_depth: float = 0.1,
        device: str = "cuda",
        num_batch: int = 1,
    ):
        """Initialize FilterDepth with configuration and pre-allocated buffers."""
        from curobo._src.util.warp import init_warp

        init_warp()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_shape = image_shape
        self.num_batch = max(int(num_batch), 1)
        H, W = image_shape
        B = self.num_batch

        # Store configuration
        self.config = FilterDepthConfig(
            depth_minimum_distance=depth_minimum_distance,
            depth_maximum_distance=depth_maximum_distance,
            flying_pixel_threshold=flying_pixel_threshold,
            bilateral_kernel_size=bilateral_kernel_size,
            bilateral_sigma_spatial=bilateral_sigma_spatial,
            bilateral_sigma_depth=bilateral_sigma_depth,
        )

        # Validate bilateral kernel size
        if bilateral_kernel_size is not None and bilateral_kernel_size % 2 == 0:
            log_and_raise(
                f"bilateral_kernel_size must be odd, got {bilateral_kernel_size}"
            )

        # Pre-compute kernel parameters
        self._setup_kernel_params()

        # Pre-allocate batched output buffers. Single-image calls use a
        # zero-copy ``unsqueeze(0)`` view into buffer[0:1].
        self._depth_out = torch.zeros((B, H, W), dtype=torch.float32, device=self.device)
        self._valid_mask_out = torch.zeros((B, H, W), dtype=torch.uint8, device=self.device)

        # Pre-allocate temp buffers for separable bilateral (if using large kernel)
        self._use_separable = (
            bilateral_kernel_size is not None and bilateral_kernel_size >= 7
        )
        if self._use_separable:
            self._depth_temp = torch.zeros(
                (B, H, W), dtype=torch.float32, device=self.device
            )
            self._depth_temp2 = torch.zeros(
                (B, H, W), dtype=torch.float32, device=self.device
            )
        else:
            self._depth_temp = None
            self._depth_temp2 = None

    def _setup_kernel_params(self):
        """Pre-compute kernel parameters for efficient execution."""
        cfg = self.config

        # Flying pixel tolerance
        if cfg.flying_pixel_threshold is not None:
            max_tol = 0.08
            min_tol = 0.005
            self._flying_tolerance = max_tol * (min_tol / max_tol) ** cfg.flying_pixel_threshold
            self._enable_flying = 1
        else:
            self._flying_tolerance = 0.0
            self._enable_flying = 0

        # Bilateral parameters
        if cfg.bilateral_kernel_size is not None:
            self._bilateral_radius = cfg.bilateral_kernel_size // 2
            self._sigma_spatial_sq2 = 2.0 * cfg.bilateral_sigma_spatial ** 2
            self._sigma_depth_sq2 = 2.0 * cfg.bilateral_sigma_depth ** 2
            self._enable_bilateral = 1
        else:
            self._bilateral_radius = 0
            self._sigma_spatial_sq2 = 1.0
            self._sigma_depth_sq2 = 1.0
            self._enable_bilateral = 0

    def __call__(
        self,
        depth_image: torch.Tensor,
        depth_out: Optional[torch.Tensor] = None,
        valid_mask_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply depth filtering.

        Args:
            depth_image: Input depth image, shape ``(B, H, W)``, float32.
                For a single image, pass ``depth.unsqueeze(0)`` and index
                ``filtered[0]`` on return.
            depth_out: Optional output buffer, shape ``(B, H, W)``. If None,
                uses pre-allocated buffer when shape matches init.
            valid_mask_out: Optional mask buffer, shape ``(B, H, W)``. If
                None, uses pre-allocated buffer when shape matches init.

        Returns:
            Tuple of (filtered_depth, valid_mask), both shape ``(B, H, W)``:
                - filtered_depth: Depth with invalid pixels set to 0
                - valid_mask: Boolean tensor where True = valid pixel

        Raises:
            ValueError: If ``depth_image`` is not 3-D.
        """
        if depth_image.dim() != 3:
            log_and_raise(
                "FilterDepth expects a batched depth tensor of shape "
                f"(B, H, W); got {tuple(depth_image.shape)}. For a single "
                "image, pass depth.unsqueeze(0)."
            )

        B, H, W = depth_image.shape
        out_depth, out_mask = self._acquire_buffers(
            B, H, W, depth_out, valid_mask_out
        )

        if self._use_separable and self._enable_bilateral:
            self._apply_separable(depth_image, out_depth, out_mask)
        else:
            self._apply_fused(depth_image, out_depth, out_mask)

        return out_depth, out_mask.bool()

    def _acquire_buffers(
        self,
        B: int,
        H: int,
        W: int,
        depth_out: Optional[torch.Tensor],
        valid_mask_out: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (B, H, W) buffers: pre-allocated if the shape matches, else fresh."""
        def _check_shape(buf: torch.Tensor, name: str):
            if buf.shape != (B, H, W):
                log_and_raise(
                    f"{name} must have shape (B, H, W)=({B}, {H}, {W}); "
                    f"got {tuple(buf.shape)}"
                )

        if depth_out is not None:
            _check_shape(depth_out, "depth_out")
        if valid_mask_out is not None:
            _check_shape(valid_mask_out, "valid_mask_out")

        shape_match = (
            H == self.image_shape[0]
            and W == self.image_shape[1]
            and B == self.num_batch
        )
        if shape_match:
            out_depth = depth_out if depth_out is not None else self._depth_out
            out_mask = (
                valid_mask_out if valid_mask_out is not None
                else self._valid_mask_out
            )
        else:
            out_depth = (
                depth_out
                if depth_out is not None
                else torch.zeros((B, H, W), dtype=torch.float32, device=self.device)
            )
            out_mask = (
                valid_mask_out
                if valid_mask_out is not None
                else torch.zeros((B, H, W), dtype=torch.uint8, device=self.device)
            )
        return out_depth, out_mask

    def _apply_fused(
        self,
        depth_in: torch.Tensor,
        depth_out: torch.Tensor,
        valid_mask_out: torch.Tensor,
    ):
        """Apply fused filter kernel on a batch of depth images."""
        import warp as wp

        from curobo._src.perception.mapper.kernel.wp_filter_depth import (
            filter_depth_fused_kernel,
        )

        B, H, W = depth_in.shape
        cfg = self.config

        check_float32_tensors(depth_in.device, depth_in=depth_in)
        wp.launch(
            filter_depth_fused_kernel,
            dim=(B, H, W),
            inputs=[
                wp.from_torch(depth_in, dtype=wp.float32),
                wp.from_torch(depth_out, dtype=wp.float32),
                wp.from_torch(valid_mask_out, dtype=wp.uint8),
                cfg.depth_minimum_distance,
                cfg.depth_maximum_distance,
                self._enable_flying,
                self._flying_tolerance,
                self._enable_bilateral,
                self._bilateral_radius,
                self._sigma_spatial_sq2,
                self._sigma_depth_sq2,
            ],
            device=str(self.device),
        )

    def _apply_separable(
        self,
        depth_in: torch.Tensor,
        depth_out: torch.Tensor,
        valid_mask_out: torch.Tensor,
    ):
        """Apply separable bilateral filter on a batch of depth images."""
        import warp as wp

        from curobo._src.perception.mapper.kernel.wp_filter_depth import (
            bilateral_filter_separable_h_kernel,
            bilateral_filter_separable_v_kernel,
            filter_depth_fused_kernel,
        )

        B, H, W = depth_in.shape
        cfg = self.config

        shape_match = (
            H == self.image_shape[0]
            and W == self.image_shape[1]
            and B == self.num_batch
            and self._depth_temp is not None
        )
        if shape_match:
            depth_temp = self._depth_temp
            depth_temp2 = self._depth_temp2
        else:
            depth_temp = torch.zeros((B, H, W), dtype=torch.float32, device=self.device)
            depth_temp2 = torch.zeros((B, H, W), dtype=torch.float32, device=self.device)

        # First pass: range + flying pixel filtering (no bilateral)
        check_float32_tensors(depth_in.device, depth_in=depth_in)
        wp.launch(
            filter_depth_fused_kernel,
            dim=(B, H, W),
            inputs=[
                wp.from_torch(depth_in, dtype=wp.float32),
                wp.from_torch(depth_temp, dtype=wp.float32),
                wp.from_torch(valid_mask_out, dtype=wp.uint8),
                cfg.depth_minimum_distance,
                cfg.depth_maximum_distance,
                self._enable_flying,
                self._flying_tolerance,
                0,  # Disable bilateral in fused kernel
                0,
                1.0,
                1.0,
            ],
            device=str(self.device),
        )

        # Second pass: separable bilateral horizontal
        wp.launch(
            bilateral_filter_separable_h_kernel,
            dim=(B, H, W),
            inputs=[
                wp.from_torch(depth_temp, dtype=wp.float32),
                wp.from_torch(depth_temp2, dtype=wp.float32),
                self._bilateral_radius,
                self._sigma_spatial_sq2,
                self._sigma_depth_sq2,
                cfg.depth_minimum_distance,
                cfg.depth_maximum_distance,
            ],
            device=str(self.device),
        )

        # Third pass: separable bilateral vertical
        wp.launch(
            bilateral_filter_separable_v_kernel,
            dim=(B, H, W),
            inputs=[
                wp.from_torch(depth_temp2, dtype=wp.float32),
                wp.from_torch(depth_out, dtype=wp.float32),
                self._bilateral_radius,
                self._sigma_spatial_sq2,
                self._sigma_depth_sq2,
                cfg.depth_minimum_distance,
                cfg.depth_maximum_distance,
            ],
            device=str(self.device),
        )

    def update_config(
        self,
        depth_minimum_distance: Optional[float] = None,
        depth_maximum_distance: Optional[float] = None,
        flying_pixel_threshold: Optional[float] = None,
        bilateral_sigma_depth: Optional[float] = None,
    ):
        """Update filter configuration without reallocating buffers.

        Only updates the specified parameters. Set to None to keep current value.

        Args:
            depth_minimum_distance: New minimum depth.
            depth_maximum_distance: New maximum depth.
            flying_pixel_threshold: New flying pixel threshold (set to 0 to disable).
            bilateral_sigma_depth: New bilateral depth sigma.
        """
        if depth_minimum_distance is not None:
            self.config.depth_minimum_distance = depth_minimum_distance
        if depth_maximum_distance is not None:
            self.config.depth_maximum_distance = depth_maximum_distance
        if flying_pixel_threshold is not None:
            if flying_pixel_threshold == 0:
                self.config.flying_pixel_threshold = None
            else:
                self.config.flying_pixel_threshold = flying_pixel_threshold
        if bilateral_sigma_depth is not None:
            self.config.bilateral_sigma_depth = bilateral_sigma_depth

        # Recompute kernel parameters
        self._setup_kernel_params()

    @classmethod
    def from_config(
        cls,
        config: FilterDepthConfig,
        image_shape: Tuple[int, int],
        device: str = "cuda",
        num_batch: int = 1,
    ) -> "FilterDepth":
        """Create FilterDepth from a configuration object.

        Args:
            config: FilterDepthConfig with filter parameters.
            image_shape: Tuple of (height, width) for depth images.
            device: CUDA device.
            num_batch: Pre-allocate buffers for this batch size.

        Returns:
            Configured FilterDepth instance.
        """
        return cls(
            image_shape=image_shape,
            depth_minimum_distance=config.depth_minimum_distance,
            depth_maximum_distance=config.depth_maximum_distance,
            flying_pixel_threshold=config.flying_pixel_threshold,
            bilateral_kernel_size=config.bilateral_kernel_size,
            bilateral_sigma_spatial=config.bilateral_sigma_spatial,
            bilateral_sigma_depth=config.bilateral_sigma_depth,
            device=device,
            num_batch=num_batch,
        )
