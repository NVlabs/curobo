# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-capacity GPU surface-point buffers and extraction workspace."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import warp as wp

from curobo._src.perception.mapper.filter_semantic import SemanticFilter
from curobo._src.perception.mapper.kernel.builder.builder_points import (
    POINT_SCAN_GROUP,
    POINT_SCAN_THREADS,
)
from curobo._src.perception.mapper.kernel.wp_voxel_extraction import _build_block_data_view
from curobo._src.perception.mapper.storage import BlockDataView, BlockSparseTSDF
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import get_warp_device_stream


@dataclass
class OccupiedPoints:
    """Borrowed GPU surface samples; only rows below ``count[0]`` are valid.

    Reused by the next extraction with the same budget. Source indices refer to
    current mapper storage and are invalid after block recycling/reset. Sampling
    does not add occupancy evidence, modify TSDF, or provide collision bounds.
    """

    #: float32 CUDA [capacity,3], world-frame positions, not voxel centers.
    positions: torch.Tensor
    #: int32 CUDA [1], number of accepted proposals (no refill).
    count: torch.Tensor
    #: int32 CUDA [capacity,2], (block pool index, local voxel index).
    source_voxels: torch.Tensor
    #: Borrowed attribute storage; do not retain across mapper mutation.
    _block_data: BlockDataView = field(repr=False)

    def colors_uint8(self) -> torch.Tensor:
        """Gather CUDA RGB [capacity,3]; only rows below the GPU count are valid."""
        rgbw = self._block_data.sample_rgbw_at_centers(self.positions, self.source_voxels[:, 0])
        rgb = (255.0 * rgbw[:, :3] / rgbw[:, 3:].clamp_min(1e-6)).clamp(0, 255)
        return torch.where(rgbw[:, 3:] > 1e-6, rgb, 128).to(torch.uint8)

    def features(self) -> torch.Tensor:
        """Gather normalized CUDA features [capacity,feature_dim] on demand."""
        if self._block_data.feature_dim == 0:
            log_and_raise("Point features require mapper feature storage")
        return self._block_data.sample_features_at_centers(
            self.positions, self.source_voxels[:, 0]
        )

    def instance_state_uint16(self) -> torch.Tensor:
        """Gather categorical owner/confidence without interpolating instance IDs."""
        return self._block_data.sample_instance_state_at_centers(
            self.positions, self.source_voxels[:, 0]
        )


class _PointBuffers:
    def __init__(self, capacity: int, tsdf: BlockSparseTSDF) -> None:
        device = tsdf.data.block_data.device
        self.result = OccupiedPoints(
            torch.zeros((capacity, 3), dtype=torch.float32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.zeros((capacity, 2), dtype=torch.int32, device=device),
            _build_block_data_view(tsdf, tsdf.config.max_blocks),
        )
        self.proposals = wp.empty((capacity, 3), dtype=wp.float32, device=str(device))
        self.sources = wp.empty((capacity, 2), dtype=wp.int32, device=str(device))
        self.valid = wp.zeros(capacity + 1, dtype=wp.int32, device=str(device))
        self.offsets = wp.zeros_like(self.valid)
        self.valid_tensor = wp.to_torch(self.valid)
        self.offsets_tensor = wp.to_torch(self.offsets)


class SurfacePointWorkspace:
    """Shared crossing index; per-budget output buffers remain alive for graphs."""

    def __init__(self, tsdf: BlockSparseTSDF) -> None:
        device = tsdf.data.block_data.device
        voxels = tsdf.config.max_blocks * tsdf.block_size**3
        if voxels > (2**31 - 1) // 3:
            log_and_raise("Surface sampling capacity exceeds int32 crossing-index limits")
        self.flags = wp.empty(
            (tsdf.config.max_blocks, tsdf.block_size**3), dtype=wp.uint8, device=str(device)
        )
        groups = (voxels + POINT_SCAN_GROUP - 1) // POINT_SCAN_GROUP
        self.counts = wp.zeros(groups + 1, dtype=wp.int32, device=str(device))
        self.offsets = wp.zeros_like(self.counts)
        self.counts_tensor = wp.to_torch(self.counts)
        self.offsets_tensor = wp.to_torch(self.offsets)
        self.query = torch.zeros(7, dtype=torch.int32, device=device)
        self.target_dummy = torch.zeros(1, device=device)
        self.selection_dummy = torch.zeros((1, 1), dtype=torch.uint8, device=device)
        self.outputs: dict[int, _PointBuffers] = {}

    def extract(
        self,
        tsdf: BlockSparseTSDF,
        sample_budget: int,
        minimum_weight: float,
        instance_slot: int | None,
        semantic_filter: SemanticFilter | None,
        voxel_selection: torch.Tensor | None,
        seed: int,
    ) -> OccupiedPoints:
        """Launch count, scans, sampling, and compaction without reading GPU counts."""
        device = tsdf.data.block_data.device
        if not math.isfinite(minimum_weight) or minimum_weight <= 0:
            log_and_raise("Surface points require a finite, positive minimum_tsdf_weight")
        if instance_slot is not None and not tsdf.data.has_instances:
            log_and_raise("instance_slot requires mapper instance storage")
        if voxel_selection is not None and (
            not isinstance(voxel_selection, torch.Tensor)
            or voxel_selection.shape != (tsdf.config.max_blocks, tsdf.block_size**3)
            or voxel_selection.dtype != torch.uint8
            or voxel_selection.device != device
            or not voxel_selection.is_contiguous()
        ):
            log_and_raise(
                "voxel_selection must be contiguous CUDA uint8 [max_blocks,block_size**3]"
            )
        if semantic_filter is not None and (
            not tsdf.data.has_features
            or semantic_filter.target.shape != (tsdf.data.feature_dim,)
            or semantic_filter.target.device != device
            or semantic_filter.target.dtype != torch.float32
            or not semantic_filter.target.is_contiguous()
        ):
            log_and_raise("semantic target must be contiguous CUDA float32 [feature_dim]")
        if sample_budget not in self.outputs:
            if torch.cuda.is_current_stream_capturing():
                log_and_raise("Warm up this sample_budget once before CUDA graph capture")
            self.outputs[sample_budget] = _PointBuffers(sample_budget, tsdf)
        buffers = self.outputs[sample_budget]
        values = [
            instance_slot if instance_slot is not None else -1,
            int(semantic_filter is not None),
            int(voxel_selection is not None),
            seed,
            int(semantic_filter is not None and semantic_filter.require_argmax),
            int(semantic_filter is not None and semantic_filter.mode == "exclude"),
            int(semantic_filter is not None and semantic_filter.min_similarity is not None),
        ]
        target = self.target_dummy if semantic_filter is None else semantic_filter.target
        selection = self.selection_dummy if voxel_selection is None else voxel_selection
        _, stream = get_warp_device_stream(tsdf.data.block_data)
        kernels = tsdf.kernels
        # All buffers are owned/updated on the caller's Torch stream. Do not add
        # a dependency on Warp's unrelated default stream during graph capture.
        with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
            wp.launch(
                kernels.set_point_query_kernel,
                dim=1,
                inputs=[wp.from_torch(self.query), *values],
                stream=stream,
            )
            self.counts.zero_()
            wp.launch(
                kernels.count_point_edges_kernel,
                dim=POINT_SCAN_THREADS,
                inputs=[
                    tsdf.get_warp_data(),
                    wp.from_torch(target),
                    wp.from_torch(selection),
                    wp.from_torch(self.query),
                    minimum_weight,
                    float(semantic_filter.min_similarity) if values[6] else -1.0,
                    self.flags,
                    self.counts,
                ],
                stream=stream,
            )
            # Torch owns scan temporaries inside Torch CUDA-graph capture. Warp's
            # utility scan allocates/frees scratch outside that capture allocator.
            torch.cumsum(
                self.counts_tensor[:-1], dim=0, dtype=torch.int32, out=self.offsets_tensor[1:]
            )
            wp.launch(
                kernels.sample_point_edges_kernel,
                dim=sample_budget,
                inputs=[
                    tsdf.get_warp_data(),
                    self.flags,
                    self.offsets,
                    wp.from_torch(self.query),
                    minimum_weight,
                    buffers.proposals,
                    buffers.sources,
                    buffers.valid,
                ],
                stream=stream,
            )
            torch.cumsum(
                buffers.valid_tensor[:-1], dim=0, dtype=torch.int32, out=buffers.offsets_tensor[1:]
            )
            wp.launch(
                kernels.compact_surface_points_kernel,
                dim=sample_budget,
                inputs=[
                    buffers.proposals,
                    buffers.sources,
                    buffers.valid,
                    buffers.offsets,
                    wp.from_torch(buffers.result.positions),
                    wp.from_torch(buffers.result.source_voxels),
                    wp.from_torch(buffers.result.count),
                ],
                stream=stream,
            )
        return buffers.result


def validate_point_query(sample_budget: int, instance_slot: int | None, seed: int) -> None:
    """Validate host metadata without synchronizing device geometry."""
    if (
        isinstance(sample_budget, bool)
        or not isinstance(sample_budget, int)
        or not 0 < sample_budget < 2**31 - 1
    ):
        log_and_raise("sample_budget must be a positive int32 capacity")
    if instance_slot is not None and (
        isinstance(instance_slot, bool)
        or not isinstance(instance_slot, int)
        or not 0 <= instance_slot <= 255
    ):
        log_and_raise("instance_slot must be an integer in [0,255]")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        log_and_raise("seed must be a nonnegative int32")
