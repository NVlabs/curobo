# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic-filter contract for block-sparse TSDF outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import warp as wp

from curobo._src.perception.mapper.kernel.warp_types import PreparedSemanticFilterWarp
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import get_warp_device_stream


@dataclass(frozen=True)
class SemanticFilter:
    """Select TSDF geometry using one fused feature target.

    The target tensor is treated as immutable. Construct a new filter after
    changing target values so prepared output masks are refreshed.

    Attributes:
        target: Target feature vector with shape ``(feature_dim,)``.
        mode: ``"only"`` retains matching geometry; ``"exclude"`` removes it.
        min_similarity: Optional inclusive cosine-similarity threshold in ``[-1, 1]``.
        require_argmax: Also require the feature and target argmax channels to match.
    """

    target: torch.Tensor
    mode: Literal["only", "exclude"] = "only"
    min_similarity: Optional[float] = None
    require_argmax: bool = False

    def __post_init__(self) -> None:
        """Validate the immutable public filter contract."""
        if not isinstance(self.target, torch.Tensor):
            log_and_raise(
                f"target must be a torch.Tensor, got {type(self.target).__name__}.",
                exception_type=TypeError,
            )
        if self.target.ndim != 1:
            log_and_raise(
                f"target must have shape (feature_dim,), got {tuple(self.target.shape)}."
            )
        if self.target.numel() == 0:
            log_and_raise("target must be non-empty.")
        if not bool(torch.isfinite(self.target).all().item()):
            log_and_raise("target must contain only finite values.")
        if self.mode not in ("only", "exclude"):
            log_and_raise(
                f"mode must be 'only' or 'exclude', got {self.mode!r}."
            )
        if not isinstance(self.require_argmax, bool):
            log_and_raise(
                f"require_argmax must be bool, got {type(self.require_argmax).__name__}.",
                exception_type=TypeError,
            )
        if self.min_similarity is None and not self.require_argmax:
            log_and_raise(
                "SemanticFilter requires at least one matching criterion: "
                "min_similarity or require_argmax=True."
            )
        if self.min_similarity is None:
            return

        threshold = float(self.min_similarity)
        if not math.isfinite(threshold) or threshold < -1.0 or threshold > 1.0:
            log_and_raise(
                "min_similarity must be finite and in [-1, 1], got "
                f"{self.min_similarity}."
            )
        if float(torch.linalg.vector_norm(self.target.float()).item()) == 0.0:
            log_and_raise("Cosine matching requires a target with nonzero norm.")


class _SemanticFilterWorkspace:
    """Own reusable semantic masks and their single-entry preparation cache."""

    _TILE_THREADS = 64

    def __init__(self) -> None:
        self.node_match: Optional[torch.Tensor] = None
        self.block_has_keep: Optional[torch.Tensor] = None
        self.target: Optional[torch.Tensor] = None
        self.warp_data: Optional[PreparedSemanticFilterWarp] = None
        self._node_revision = -1
        self._node_target: Optional[torch.Tensor] = None
        self._node_min_similarity: Optional[float] = None
        self._node_require_argmax = False
        self._summary_mode: Optional[str] = None

    def _allocate(self, tsdf: object) -> None:
        device = tsdf.data.block_features.device
        self.node_match = torch.empty(
            (
                tsdf.config.max_blocks,
                tsdf.data.feature_block_grid_size**3,
            ),
            dtype=torch.uint8,
            device=device,
        )
        self.block_has_keep = torch.empty(
            tsdf.config.max_blocks,
            dtype=torch.uint8,
            device=device,
        )
        self.target = torch.empty(
            tsdf.data.feature_dim,
            dtype=torch.float32,
            device=device,
        )
        warp_data = PreparedSemanticFilterWarp()
        warp_data.node_match = wp.from_torch(self.node_match, dtype=wp.uint8)
        warp_data.block_has_keep = wp.from_torch(self.block_has_keep, dtype=wp.uint8)
        warp_data.exclude = False
        self.warp_data = warp_data

    @staticmethod
    def _validate_for_tsdf(semantic_filter: SemanticFilter, tsdf: object) -> None:
        if not tsdf.data.has_features:
            log_and_raise(
                "Semantic filtering requires mapper features; configure feature_dim > 0."
            )
        expected_dim = int(tsdf.data.feature_dim)
        if semantic_filter.target.shape[0] != expected_dim:
            log_and_raise(
                f"SemanticFilter.target must have shape ({expected_dim},), got "
                f"{tuple(semantic_filter.target.shape)}."
            )

    def prepare(
        self,
        semantic_filter: SemanticFilter,
        tsdf: object,
        semantic_revision: int,
    ) -> PreparedSemanticFilterWarp:
        """Prepare and return a cached Warp filter view for the current map."""
        self._validate_for_tsdf(semantic_filter, tsdf)
        if self.warp_data is None:
            self._allocate(tsdf)
        assert self.node_match is not None
        assert self.block_has_keep is not None
        assert self.target is not None
        assert self.warp_data is not None

        node_cache_hit = (
            self._node_revision == semantic_revision
            and self._node_target is semantic_filter.target
            and self._node_min_similarity == semantic_filter.min_similarity
            and self._node_require_argmax == semantic_filter.require_argmax
        )
        device, stream = get_warp_device_stream(self.target)
        if not node_cache_hit:
            target = semantic_filter.target.detach().to(
                device=self.target.device,
                dtype=torch.float32,
            )
            if not target.is_contiguous():
                target = target.contiguous()
            if semantic_filter.min_similarity is not None:
                target = target / torch.linalg.vector_norm(target)
            self.target.copy_(target)
            target_argmax = (
                int(torch.argmax(target).item())
                if semantic_filter.require_argmax
                else 0
            )
            wp.launch_tiled(
                tsdf.kernels.prepare_semantic_node_matches_kernel,
                dim=(
                    tsdf.config.max_blocks,
                    tsdf.data.feature_block_grid_size**3,
                ),
                block_dim=self._TILE_THREADS,
                inputs=[
                    tsdf.get_warp_data(),
                    wp.from_torch(self.target, dtype=wp.float32),
                    (
                        float(semantic_filter.min_similarity)
                        if semantic_filter.min_similarity is not None
                        else -1.0
                    ),
                    target_argmax,
                    1 if semantic_filter.min_similarity is not None else 0,
                    1 if semantic_filter.require_argmax else 0,
                    wp.from_torch(self.node_match, dtype=wp.uint8),
                ],
                device=device,
                stream=stream,
            )
            self._node_revision = semantic_revision
            self._node_target = semantic_filter.target
            self._node_min_similarity = semantic_filter.min_similarity
            self._node_require_argmax = semantic_filter.require_argmax
            self._summary_mode = None

        if self._summary_mode != semantic_filter.mode:
            wp.launch_tiled(
                tsdf.kernels.prepare_semantic_block_summary_kernel,
                dim=(tsdf.config.max_blocks,),
                block_dim=self._TILE_THREADS,
                inputs=[
                    tsdf.get_warp_data(),
                    wp.from_torch(self.node_match, dtype=wp.uint8),
                    1 if semantic_filter.mode == "exclude" else 0,
                    wp.from_torch(self.block_has_keep, dtype=wp.uint8),
                ],
                device=device,
                stream=stream,
            )
            self._summary_mode = semantic_filter.mode

        self.warp_data.exclude = semantic_filter.mode == "exclude"
        return self.warp_data
