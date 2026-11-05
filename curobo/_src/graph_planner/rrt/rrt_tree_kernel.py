# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Graph-capturable Warp kernels for batched packed RRT trees.

The nearest-query boundary stops at nearest starts and parent indices. The
shared RRT Warp kernels consume ``binding.starts`` and ``binding.desired``
afterward so one implementation controls extension limiting and collision
sampling in both Python and native runtimes.

Bindings retain Torch-owned tensors and cache zero-copy Warp array views. The
tensor addresses and shapes must remain stable for the lifetime of a binding;
their contents, including ``tree_counts``, may change before every graph replay.
"""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import (
    get_warp_device_stream,
    warp_constant_suffix,
    warp_func,
    warp_kernel,
)

__all__ = [
    "RRTTreeKernelRunner",
    "NearestRRTTreeQueryBuffers",
    "PackedRRTTreeAppendBuffers",
    "RRTPathReconstructionBuffers",
    "RRTTreeKernelConfig",
]


@dataclass(frozen=True)
class RRTTreeKernelConfig:
    """Compile-time constants for one cached batched-tree kernel bundle."""

    action_dim: int
    max_candidates: int
    max_path_nodes: int = 68
    block_dim: int = 64

    def __post_init__(self) -> None:
        """Resolve a scan width that covers the requested candidate frontier."""
        if self.action_dim <= 0:
            log_and_raise("action_dim must be positive")
        if self.max_candidates <= 0:
            log_and_raise("max_candidates must be positive")
        if self.max_path_nodes < 2:
            log_and_raise("max_path_nodes must be at least two")
        block_dim = max(32, int(self.block_dim))
        while block_dim < self.max_candidates:
            block_dim *= 2
        if block_dim > 256:
            log_and_raise("generated RRT Warp kernels support at most 256 candidates")
        if block_dim & (block_dim - 1):
            log_and_raise("block_dim must be a power of two")
        object.__setattr__(self, "block_dim", block_dim)


@dataclass(frozen=True)
class _RRTTreeKernelBundle:
    nearest_target: Any
    packed_append: Any
    reconstruct_path: Any


@lru_cache(maxsize=None)
def _make_tree_kernels(
    action_dim: int,
    block_dim: int,
    max_path_nodes: int,
) -> _RRTTreeKernelBundle:
    """Generate one DoF/scan-width-specialized batched Warp module.

    Kernel names include a stable encoding of constants. Warp and NVRTC therefore
    reuse the on-disk module cache when a robot/profile is constructed again.
    """
    dof = wp.constant(action_dim)
    nearest_chunk = wp.constant(block_dim)
    scan_width = wp.constant(block_dim)
    output_chunks = wp.constant((action_dim + block_dim - 1) // block_dim)
    path_capacity = wp.constant(max_path_nodes)
    suffix = warp_constant_suffix("rrt_batched_v9", action_dim, block_dim, max_path_nodes)

    @warp_func(f"rrt_mask_outside_tree_count_{suffix}")
    def mask_outside_tree_count(distance: float, node_index: int, tree_count: int) -> float:
        if node_index < tree_count:
            return distance
        return float(3.402823466e38)

    @warp_kernel(
        f"rrt_batched_packed_nearest_kernel_{suffix}",
        enable_backward=False,
    )
    def batched_packed_nearest_kernel(
        tree_nodes: wp.array2d(dtype=wp.float32),
        tree_counts: wp.array(dtype=wp.int32),
        desired: wp.array2d(dtype=wp.float32),
        distance_weight: wp.array(dtype=wp.float32),
        starts: wp.array2d(dtype=wp.float32),
        parent_indices: wp.array(dtype=wp.int64),
    ):
        flat_query, lane = wp.tid()
        batch_count = tree_counts.shape[0]
        queries_per_batch = desired.shape[0] // batch_count
        tree_capacity = tree_nodes.shape[0] // batch_count
        batch_index = flat_query // queries_per_batch
        tree_offset = batch_index * tree_capacity
        count = wp.clamp(
            tree_counts[batch_index],
            wp.int32(1),
            wp.int32(tree_capacity),
        )
        best_distance = float(3.402823466e38)
        best_index = int(0)
        chunk_count = int((count + nearest_chunk - 1) / nearest_chunk)

        for chunk in range(chunk_count):
            base = chunk * nearest_chunk
            node_tile = wp.tile_load(
                tree_nodes,
                shape=(nearest_chunk, dof),
                offset=(tree_offset + base, 0),
                bounds_check=True,
            )
            desired_row = wp.tile_load(desired, shape=(1, dof), offset=(flat_query, 0))
            weight_row = wp.tile_load(distance_weight, shape=dof, offset=0)
            desired_tile = wp.tile_broadcast(desired_row, shape=(nearest_chunk, dof))
            weight_tile = wp.tile_broadcast(weight_row, shape=(nearest_chunk, dof))
            delta = (desired_tile - node_tile) * weight_tile
            distances = wp.tile_reduce(wp.add, delta * delta, axis=1)
            node_indices = wp.tile_arange(nearest_chunk, dtype=int) + wp.tile_full(
                nearest_chunk,
                value=base,
                dtype=int,
            )
            distances = wp.tile_map(
                mask_outside_tree_count,
                distances,
                node_indices,
                count,
            )
            local_index_tile = wp.tile_argmin(distances)
            local_min_tile = wp.tile_min(distances)
            local_index = wp.tile_extract(local_index_tile, 0)
            local_min = wp.tile_extract(local_min_tile, 0)
            if local_min < best_distance:
                best_distance = local_min
                best_index = base + local_index

        global_best_index = tree_offset + best_index
        if lane == 0:
            parent_indices[flat_query] = wp.int64(best_index)
        for output_chunk in range(output_chunks):
            joint = lane + output_chunk * scan_width
            if joint < dof:
                starts[flat_query, joint] = tree_nodes[global_best_index, joint]

    @warp_kernel(
        f"rrt_batched_packed_append_kernel_{suffix}",
        enable_backward=False,
    )
    def batched_packed_append_kernel(
        tree_nodes: wp.array2d(dtype=wp.float32),
        tree_parents: wp.array(dtype=wp.int64),
        tree_counts: wp.array(dtype=wp.int32),
        candidates: wp.array2d(dtype=wp.float32),
        candidate_parents: wp.array(dtype=wp.int64),
        candidate_valid: wp.array(dtype=wp.bool),
        assigned_slots: wp.array(dtype=wp.int64),
        write_mask: wp.array(dtype=wp.bool),
    ):
        batch_index, lane = wp.tid()
        batch_count = tree_counts.shape[0]
        candidate_count = candidates.shape[0] // batch_count
        tree_capacity = tree_nodes.shape[0] // batch_count
        candidate_offset = batch_index * candidate_count
        candidate_index = candidate_offset + lane
        tree_offset = batch_index * tree_capacity
        base = wp.clamp(
            tree_counts[batch_index],
            wp.int32(0),
            wp.int32(tree_capacity),
        )
        valid_value = int(0)
        candidate_is_valid = False
        if lane < candidate_count:
            parent = candidate_parents[candidate_index]
            parent_is_valid = parent >= wp.int64(0) and parent < wp.int64(base)
            candidate_is_valid = candidate_valid[candidate_index] and parent_is_valid
            if candidate_is_valid:
                valid_value = 1

        valid_tile = wp.tile(valid_value)
        inclusive_rank = wp.tile_scan_inclusive(valid_tile)
        total_valid = wp.tile_extract(inclusive_rank, scan_width - 1)

        if lane < candidate_count:
            rank = wp.tile_extract(inclusive_rank, lane) - 1
            slot = base + rank
            can_write = candidate_is_valid and slot >= 0 and slot < tree_capacity
            assigned_slots[candidate_index] = wp.int64(slot)
            write_mask[candidate_index] = can_write
            if can_write:
                global_slot = tree_offset + slot
                for joint in range(dof):
                    tree_nodes[global_slot, joint] = candidates[candidate_index, joint]
                tree_parents[global_slot] = candidate_parents[candidate_index]

        if lane == 0:
            tree_counts[batch_index] = wp.min(base + total_valid, tree_capacity)

    @warp_kernel(
        f"rrt_batched_reconstruct_path_kernel_{suffix}",
        enable_backward=False,
    )
    def batched_reconstruct_path_kernel(
        start_nodes: wp.array2d(dtype=wp.float32),
        start_parents: wp.array(dtype=wp.int64),
        goal_nodes: wp.array2d(dtype=wp.float32),
        goal_parents: wp.array(dtype=wp.int64),
        connection_indices: wp.array(dtype=wp.int64),
        distance_weight: wp.array(dtype=wp.float32),
        output_paths: wp.array2d(dtype=wp.float32),
        path_counts: wp.array(dtype=wp.int32),
        path_lengths: wp.array(dtype=wp.float32),
    ):
        batch_index = wp.tid()
        batch_count = path_counts.shape[0]
        tree_capacity = start_parents.shape[0] // batch_count
        tree_offset = batch_index * tree_capacity
        output_offset = batch_index * path_capacity
        metadata_offset = batch_index * 2
        start_index = int(connection_indices[metadata_offset])
        goal_index = int(connection_indices[metadata_offset + 1])
        if start_index == -2 and goal_index == -2:
            length_squared = float(0.0)
            for joint in range(dof):
                start_value = start_nodes[tree_offset, joint]
                goal_value = goal_nodes[tree_offset, joint]
                output_paths[output_offset, joint] = start_value
                output_paths[output_offset + 1, joint] = goal_value
                delta = (goal_value - start_value) * distance_weight[joint]
                length_squared = length_squared + delta * delta
            path_counts[batch_index] = wp.int32(2)
            path_lengths[batch_index] = wp.sqrt(length_squared)
            return
        if start_index < 0 or goal_index < 0:
            path_counts[batch_index] = wp.int32(0)
            path_lengths[batch_index] = wp.float32(wp.inf)
            return

        start_depth = int(0)
        start_cursor = start_index
        invalid = bool(False)
        for _ in range(path_capacity):
            if start_cursor >= 0 and not invalid:
                if start_cursor >= tree_capacity:
                    invalid = True
                else:
                    start_depth = start_depth + 1
                    start_cursor = int(start_parents[tree_offset + start_cursor])
        if start_cursor >= 0:
            invalid = True

        goal_depth = int(0)
        goal_cursor = goal_index
        for _ in range(path_capacity):
            if goal_cursor >= 0 and not invalid:
                if goal_cursor >= tree_capacity:
                    invalid = True
                else:
                    goal_depth = goal_depth + 1
                    goal_cursor = int(goal_parents[tree_offset + goal_cursor])
        if goal_cursor >= 0:
            invalid = True

        connection_distance_squared = float(0.0)
        if not invalid:
            for joint in range(dof):
                delta = (
                    start_nodes[tree_offset + start_index, joint]
                    - goal_nodes[tree_offset + goal_index, joint]
                )
                connection_distance_squared = connection_distance_squared + delta * delta
        duplicate_connection = connection_distance_squared <= 1.0e-14
        output_count = start_depth + goal_depth
        if duplicate_connection:
            output_count = output_count - 1
        if invalid or output_count < 2 or output_count > path_capacity:
            path_counts[batch_index] = wp.int32(-1)
            path_lengths[batch_index] = wp.float32(wp.inf)
            return

        start_cursor = start_index
        for step in range(path_capacity):
            if step < start_depth:
                output_index = start_depth - 1 - step
                for joint in range(dof):
                    output_paths[output_offset + output_index, joint] = start_nodes[
                        tree_offset + start_cursor, joint
                    ]
                start_cursor = int(start_parents[tree_offset + start_cursor])

        goal_cursor = goal_index
        goal_skip = int(0)
        if duplicate_connection:
            goal_skip = 1
        for step in range(path_capacity):
            if step < goal_depth:
                if step >= goal_skip:
                    output_index = start_depth + step - goal_skip
                    for joint in range(dof):
                        output_paths[output_offset + output_index, joint] = goal_nodes[
                            tree_offset + goal_cursor, joint
                        ]
                goal_cursor = int(goal_parents[tree_offset + goal_cursor])
        path_counts[batch_index] = wp.int32(output_count)
        path_length = float(0.0)
        for step in range(path_capacity - 1):
            if step + 1 < output_count:
                edge_length_squared = float(0.0)
                for joint in range(dof):
                    delta = (
                        output_paths[output_offset + step + 1, joint]
                        - output_paths[output_offset + step, joint]
                    ) * distance_weight[joint]
                    edge_length_squared = edge_length_squared + delta * delta
                path_length = path_length + wp.sqrt(edge_length_squared)
        path_lengths[batch_index] = path_length

    # Match the production rollout's arithmetic and launch defaults. These
    # options must be applied before the generated module is first compiled.
    for kernel in (
        batched_packed_nearest_kernel,
        batched_packed_append_kernel,
        batched_reconstruct_path_kernel,
    ):
        kernel.module.options.update(
            {
                "enable_backward": False,
                "block_dim": block_dim,
                "fuse_fp": False,
            }
        )
    return _RRTTreeKernelBundle(
        batched_packed_nearest_kernel,
        batched_packed_append_kernel,
        batched_reconstruct_path_kernel,
    )


def _validate_cuda_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    ndim: int,
    device: torch.device | None = None,
) -> torch.device:
    if not isinstance(tensor, torch.Tensor):
        log_and_raise(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        log_and_raise(f"{name} must use {dtype}, received {tensor.dtype}")
    if tensor.ndim != ndim:
        log_and_raise(f"{name} must have {ndim} dimensions, received {tensor.ndim}")
    if tensor.device.type != "cuda":
        log_and_raise(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        log_and_raise(f"{name} must be contiguous")
    if device is not None and tensor.device != device:
        log_and_raise(f"{name} must be on {device}, received {tensor.device}")
    return tensor.device


def _warp_views(**tensors: torch.Tensor) -> dict[str, Any]:
    """Create persistent zero-copy Warp views for Torch-owned buffers."""
    wp.init()
    return {name: wp.from_torch(tensor) for name, tensor in tensors.items()}


@dataclass
class NearestRRTTreeQueryBuffers:
    """Persistent buffers for nearest queries over ``B`` packed trees."""

    profile: RRTTreeKernelConfig
    tree_nodes: torch.Tensor
    tree_counts: torch.Tensor
    desired: torch.Tensor
    distance_weight: torch.Tensor
    starts: torch.Tensor
    parent_indices: torch.Tensor
    _flat_tensors: dict[str, torch.Tensor] = field(init=False, repr=False)
    _views: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate ``[B,N,D]``/``[B,K,D]`` storage and cache flat views."""
        device = _validate_cuda_tensor("tree_nodes", self.tree_nodes, dtype=torch.float32, ndim=3)
        batch_size, capacity, action_dim = self.tree_nodes.shape
        if batch_size < 1 or capacity < 1 or action_dim != self.profile.action_dim:
            log_and_raise(
                f"tree_nodes must have nonempty shape (batch, capacity, {self.profile.action_dim})"
            )
        if capacity % self.profile.block_dim != 0:
            log_and_raise("batched tree capacity must be a multiple of profile.block_dim")
        _validate_cuda_tensor(
            "tree_counts", self.tree_counts, dtype=torch.int32, ndim=1, device=device
        )
        if self.tree_counts.shape != (batch_size,):
            log_and_raise("tree_counts must have shape (batch,)")
        _validate_cuda_tensor("desired", self.desired, dtype=torch.float32, ndim=3, device=device)
        if (
            self.desired.shape[0] != batch_size
            or self.desired.shape[2] != action_dim
            or not 1 <= self.desired.shape[1] <= self.profile.max_candidates
        ):
            log_and_raise(
                "desired must have shape (batch, candidates, action_dim) within the profile"
            )
        _validate_cuda_tensor(
            "distance_weight",
            self.distance_weight,
            dtype=torch.float32,
            ndim=1,
            device=device,
        )
        if self.distance_weight.shape != (action_dim,):
            log_and_raise("distance_weight must have shape (action_dim,)")
        for name, tensor, dtype, ndim in (
            ("starts", self.starts, torch.float32, 3),
            ("parent_indices", self.parent_indices, torch.int64, 2),
        ):
            _validate_cuda_tensor(name, tensor, dtype=dtype, ndim=ndim, device=device)
        if self.starts.shape != self.desired.shape:
            log_and_raise("starts must match desired.shape")
        if self.parent_indices.shape != self.desired.shape[:2]:
            log_and_raise("parent_indices must have shape (batch, candidates)")
        self._flat_tensors = {
            "tree_nodes": self.tree_nodes.view(batch_size * capacity, action_dim),
            "desired": self.desired.view(-1, action_dim),
            "starts": self.starts.view(-1, action_dim),
            "parent_indices": self.parent_indices.view(-1),
        }
        self._views = _warp_views(
            tree_nodes=self._flat_tensors["tree_nodes"],
            tree_counts=self.tree_counts,
            desired=self._flat_tensors["desired"],
            distance_weight=self.distance_weight,
            starts=self._flat_tensors["starts"],
            parent_indices=self._flat_tensors["parent_indices"],
        )

    @property
    def device(self) -> torch.device:
        """Return the CUDA device shared by all batched nearest tensors."""
        return self.tree_nodes.device


@dataclass
class PackedRRTTreeAppendBuffers:
    """Persistent deterministic append buffers for ``B`` packed trees."""

    profile: RRTTreeKernelConfig
    tree_nodes: torch.Tensor
    tree_parents: torch.Tensor
    tree_counts: torch.Tensor
    candidates: torch.Tensor
    candidate_parents: torch.Tensor
    candidate_valid: torch.Tensor
    assigned_slots: torch.Tensor
    write_mask: torch.Tensor
    _flat_tensors: dict[str, torch.Tensor] = field(init=False, repr=False)
    _views: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the batched packed-tree ABI and cache flat Warp views."""
        device = _validate_cuda_tensor("tree_nodes", self.tree_nodes, dtype=torch.float32, ndim=3)
        batch_size, capacity, action_dim = self.tree_nodes.shape
        if batch_size < 1 or capacity < 1 or action_dim != self.profile.action_dim:
            log_and_raise(
                f"tree_nodes must have nonempty shape (batch, capacity, {self.profile.action_dim})"
            )
        if capacity % self.profile.block_dim != 0:
            log_and_raise("batched tree capacity must be a multiple of profile.block_dim")
        _validate_cuda_tensor(
            "tree_parents", self.tree_parents, dtype=torch.int64, ndim=2, device=device
        )
        if self.tree_parents.shape != (batch_size, capacity):
            log_and_raise("tree_parents must have shape (batch, capacity)")
        _validate_cuda_tensor(
            "tree_counts", self.tree_counts, dtype=torch.int32, ndim=1, device=device
        )
        if self.tree_counts.shape != (batch_size,):
            log_and_raise("tree_counts must have shape (batch,)")
        _validate_cuda_tensor(
            "candidates", self.candidates, dtype=torch.float32, ndim=3, device=device
        )
        candidate_shape = self.candidates.shape[:2]
        if (
            candidate_shape[0] != batch_size
            or self.candidates.shape[2] != action_dim
            or not 1 <= candidate_shape[1] <= self.profile.max_candidates
        ):
            log_and_raise(
                "candidates must have shape (batch, candidates, action_dim) within the profile"
            )
        for name, tensor, dtype in (
            ("candidate_parents", self.candidate_parents, torch.int64),
            ("candidate_valid", self.candidate_valid, torch.bool),
            ("assigned_slots", self.assigned_slots, torch.int64),
            ("write_mask", self.write_mask, torch.bool),
        ):
            _validate_cuda_tensor(name, tensor, dtype=dtype, ndim=2, device=device)
            if tensor.shape != candidate_shape:
                log_and_raise(f"{name} must have shape (batch, candidates)")
        self._flat_tensors = {
            "tree_nodes": self.tree_nodes.view(batch_size * capacity, action_dim),
            "tree_parents": self.tree_parents.view(-1),
            "candidates": self.candidates.view(-1, action_dim),
            "candidate_parents": self.candidate_parents.view(-1),
            "candidate_valid": self.candidate_valid.view(-1),
            "assigned_slots": self.assigned_slots.view(-1),
            "write_mask": self.write_mask.view(-1),
        }
        self._views = _warp_views(
            tree_nodes=self._flat_tensors["tree_nodes"],
            tree_parents=self._flat_tensors["tree_parents"],
            tree_counts=self.tree_counts,
            candidates=self._flat_tensors["candidates"],
            candidate_parents=self._flat_tensors["candidate_parents"],
            candidate_valid=self._flat_tensors["candidate_valid"],
            assigned_slots=self._flat_tensors["assigned_slots"],
            write_mask=self._flat_tensors["write_mask"],
        )

    @property
    def device(self) -> torch.device:
        """Return the CUDA device shared by all batched append tensors."""
        return self.tree_nodes.device


@dataclass
class RRTPathReconstructionBuffers:
    """Persistent buffers for batched GPU parent-chain reconstruction."""

    profile: RRTTreeKernelConfig
    start_nodes: torch.Tensor
    start_parents: torch.Tensor
    goal_nodes: torch.Tensor
    goal_parents: torch.Tensor
    connection_indices: torch.Tensor
    distance_weight: torch.Tensor
    output_paths: torch.Tensor
    path_counts: torch.Tensor
    path_lengths: torch.Tensor
    _flat_tensors: dict[str, torch.Tensor] = field(init=False, repr=False)
    _views: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate tree storage and fixed reconstruction output capacity."""
        device = _validate_cuda_tensor(
            "start_nodes", self.start_nodes, dtype=torch.float32, ndim=3
        )
        batch_size, tree_capacity, action_dim = self.start_nodes.shape
        if batch_size < 1 or tree_capacity < 1 or action_dim != self.profile.action_dim:
            log_and_raise("start_nodes must have nonempty shape (batch, capacity, action_dim)")
        _validate_cuda_tensor(
            "goal_nodes", self.goal_nodes, dtype=torch.float32, ndim=3, device=device
        )
        if self.goal_nodes.shape != self.start_nodes.shape:
            log_and_raise("goal_nodes must match start_nodes.shape")
        for name, parents in (
            ("start_parents", self.start_parents),
            ("goal_parents", self.goal_parents),
        ):
            _validate_cuda_tensor(name, parents, dtype=torch.int64, ndim=2, device=device)
            if parents.shape != (batch_size, tree_capacity):
                log_and_raise(f"{name} must have shape (batch, tree_capacity)")
        _validate_cuda_tensor(
            "connection_indices",
            self.connection_indices,
            dtype=torch.int64,
            ndim=2,
            device=device,
        )
        if self.connection_indices.shape != (batch_size, 2):
            log_and_raise("connection_indices must have shape (batch, 2)")
        _validate_cuda_tensor(
            "distance_weight",
            self.distance_weight,
            dtype=torch.float32,
            ndim=1,
            device=device,
        )
        if self.distance_weight.shape != (action_dim,):
            log_and_raise("distance_weight must have shape (action_dim,)")
        _validate_cuda_tensor(
            "output_paths", self.output_paths, dtype=torch.float32, ndim=3, device=device
        )
        if self.output_paths.shape != (
            batch_size,
            self.profile.max_path_nodes,
            action_dim,
        ):
            log_and_raise("output_paths must match the profile reconstruction capacity")
        _validate_cuda_tensor(
            "path_counts", self.path_counts, dtype=torch.int32, ndim=1, device=device
        )
        if self.path_counts.shape != (batch_size,):
            log_and_raise("path_counts must have shape (batch,)")
        _validate_cuda_tensor(
            "path_lengths", self.path_lengths, dtype=torch.float32, ndim=1, device=device
        )
        if self.path_lengths.shape != (batch_size,):
            log_and_raise("path_lengths must have shape (batch,)")
        self._flat_tensors = {
            "start_nodes": self.start_nodes.view(batch_size * tree_capacity, action_dim),
            "start_parents": self.start_parents.view(-1),
            "goal_nodes": self.goal_nodes.view(batch_size * tree_capacity, action_dim),
            "goal_parents": self.goal_parents.view(-1),
            "connection_indices": self.connection_indices.view(-1),
            "output_paths": self.output_paths.view(-1, action_dim),
        }
        self._views = _warp_views(
            start_nodes=self._flat_tensors["start_nodes"],
            start_parents=self._flat_tensors["start_parents"],
            goal_nodes=self._flat_tensors["goal_nodes"],
            goal_parents=self._flat_tensors["goal_parents"],
            connection_indices=self._flat_tensors["connection_indices"],
            distance_weight=self.distance_weight,
            output_paths=self._flat_tensors["output_paths"],
            path_counts=self.path_counts,
            path_lengths=self.path_lengths,
        )

    @property
    def device(self) -> torch.device:
        """Return the CUDA device shared by reconstruction tensors."""
        return self.start_nodes.device


class RRTTreeKernelRunner:
    """Bind persistent Torch buffers and launch the batched graph-safe kernels."""

    def __init__(self, profile: RRTTreeKernelConfig) -> None:
        """Initialize or reuse a generated profile without planner buffers."""
        wp.init()
        self.profile = profile
        self._kernels = _make_tree_kernels(
            self.profile.action_dim,
            self.profile.block_dim,
            self.profile.max_path_nodes,
        )
        self._loaded_devices: set[str] = set()

    def load_modules(self, device: torch.device) -> None:
        """Compile/load the profile module before latency-sensitive capture."""
        device_key = str(device)
        if device_key in self._loaded_devices:
            return
        for kernel in (
            self._kernels.nearest_target,
            self._kernels.packed_append,
            self._kernels.reconstruct_path,
        ):
            wp.load_module(
                module=kernel.module,
                device=device_key,
                block_dim=self.profile.block_dim,
            )
        self._loaded_devices.add(device_key)

    def bind_batched_nearest_target(
        self,
        tree_nodes: torch.Tensor,
        tree_counts: torch.Tensor,
        desired: torch.Tensor,
        distance_weight: torch.Tensor,
        *,
        starts: torch.Tensor | None = None,
        parent_indices: torch.Tensor | None = None,
    ) -> NearestRRTTreeQueryBuffers:
        """Bind ``[B,N,D]`` trees and ``[B,K,D]`` nearest queries."""
        starts = torch.empty_like(desired) if starts is None else starts
        parent_indices = (
            torch.empty(desired.shape[:2], dtype=torch.int64, device=desired.device)
            if parent_indices is None
            else parent_indices
        )
        return NearestRRTTreeQueryBuffers(
            profile=self.profile,
            tree_nodes=tree_nodes,
            tree_counts=tree_counts,
            desired=desired,
            distance_weight=distance_weight,
            starts=starts,
            parent_indices=parent_indices,
        )

    def bind_batched_packed_append(
        self,
        tree_nodes: torch.Tensor,
        tree_parents: torch.Tensor,
        tree_counts: torch.Tensor,
        candidates: torch.Tensor,
        candidate_parents: torch.Tensor,
        candidate_valid: torch.Tensor,
        *,
        assigned_slots: torch.Tensor | None = None,
        write_mask: torch.Tensor | None = None,
    ) -> PackedRRTTreeAppendBuffers:
        """Bind deterministic packed appends for independent batched trees."""
        candidate_shape = candidates.shape[:2]
        assigned_slots = (
            torch.empty(candidate_shape, dtype=torch.int64, device=candidates.device)
            if assigned_slots is None
            else assigned_slots
        )
        write_mask = (
            torch.empty(candidate_shape, dtype=torch.bool, device=candidates.device)
            if write_mask is None
            else write_mask
        )
        return PackedRRTTreeAppendBuffers(
            profile=self.profile,
            tree_nodes=tree_nodes,
            tree_parents=tree_parents,
            tree_counts=tree_counts,
            candidates=candidates,
            candidate_parents=candidate_parents,
            candidate_valid=candidate_valid,
            assigned_slots=assigned_slots,
            write_mask=write_mask,
        )

    def bind_batched_path_reconstruction(
        self,
        start_nodes: torch.Tensor,
        start_parents: torch.Tensor,
        goal_nodes: torch.Tensor,
        goal_parents: torch.Tensor,
        distance_weight: torch.Tensor,
        *,
        connection_indices: torch.Tensor | None = None,
        output_paths: torch.Tensor | None = None,
        path_counts: torch.Tensor | None = None,
        path_lengths: torch.Tensor | None = None,
    ) -> RRTPathReconstructionBuffers:
        """Bind two packed trees to one fixed-capacity batched path output."""
        batch_size, _, action_dim = start_nodes.shape
        connection_indices = (
            torch.full((batch_size, 2), -1, dtype=torch.int64, device=start_nodes.device)
            if connection_indices is None
            else connection_indices
        )
        output_paths = (
            torch.empty(
                (batch_size, self.profile.max_path_nodes, action_dim),
                dtype=start_nodes.dtype,
                device=start_nodes.device,
            )
            if output_paths is None
            else output_paths
        )
        path_counts = (
            torch.zeros(batch_size, dtype=torch.int32, device=start_nodes.device)
            if path_counts is None
            else path_counts
        )
        path_lengths = (
            torch.full(
                (batch_size,),
                torch.inf,
                dtype=start_nodes.dtype,
                device=start_nodes.device,
            )
            if path_lengths is None
            else path_lengths
        )
        return RRTPathReconstructionBuffers(
            profile=self.profile,
            start_nodes=start_nodes,
            start_parents=start_parents,
            goal_nodes=goal_nodes,
            goal_parents=goal_parents,
            connection_indices=connection_indices,
            distance_weight=distance_weight,
            output_paths=output_paths,
            path_counts=path_counts,
            path_lengths=path_lengths,
        )

    @staticmethod
    def _resolve_stream(
        device: torch.device,
        stream: wp.Stream | torch.cuda.Stream | None,
    ) -> wp.Stream:
        if isinstance(stream, torch.cuda.Stream):
            with torch.cuda.stream(stream):
                _, warp_stream = get_warp_device_stream(device)
            if warp_stream is None:
                log_and_raise("RRT Warp kernels require a CUDA stream")
            return warp_stream
        if stream is not None:
            return stream
        _, warp_stream = get_warp_device_stream(device)
        if warp_stream is None:
            log_and_raise("RRT Warp kernels require a CUDA stream")
        return warp_stream

    def batched_nearest_target(
        self,
        binding: NearestRRTTreeQueryBuffers,
        *,
        stream: wp.Stream | torch.cuda.Stream | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Launch nearest queries for every packed-tree batch lane."""
        if binding.profile != self.profile:
            log_and_raise("batched nearest binding belongs to a different profile")
        warp_stream = self._resolve_stream(binding.device, stream)
        wp.launch_tiled(
            self._kernels.nearest_target,
            dim=(binding.desired.shape[0] * binding.desired.shape[1],),
            inputs=[
                binding._views["tree_nodes"],
                binding._views["tree_counts"],
                binding._views["desired"],
                binding._views["distance_weight"],
            ],
            outputs=[
                binding._views["starts"],
                binding._views["parent_indices"],
            ],
            block_dim=self.profile.block_dim,
            stream=warp_stream,
        )
        return binding.starts, binding.parent_indices

    def batched_packed_append(
        self,
        binding: PackedRRTTreeAppendBuffers,
        *,
        stream: wp.Stream | torch.cuda.Stream | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append candidates independently for every packed-tree batch lane."""
        if binding.profile != self.profile:
            log_and_raise("batched append binding belongs to a different profile")
        warp_stream = self._resolve_stream(binding.device, stream)
        wp.launch_tiled(
            self._kernels.packed_append,
            dim=(binding.tree_nodes.shape[0],),
            inputs=[
                binding._views["tree_nodes"],
                binding._views["tree_parents"],
                binding._views["tree_counts"],
                binding._views["candidates"],
                binding._views["candidate_parents"],
                binding._views["candidate_valid"],
            ],
            outputs=[
                binding._views["assigned_slots"],
                binding._views["write_mask"],
            ],
            block_dim=self.profile.block_dim,
            stream=warp_stream,
        )
        return binding.assigned_slots, binding.write_mask, binding.tree_counts

    def batched_reconstruct_path(
        self,
        binding: RRTPathReconstructionBuffers,
        *,
        stream: wp.Stream | torch.cuda.Stream | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trace every active connection into start-to-goal GPU waypoint storage."""
        if binding.profile != self.profile:
            log_and_raise("path reconstruction binding belongs to a different profile")
        warp_stream = self._resolve_stream(binding.device, stream)
        wp.launch(
            self._kernels.reconstruct_path,
            dim=binding.start_nodes.shape[0],
            inputs=[
                binding._views["start_nodes"],
                binding._views["start_parents"],
                binding._views["goal_nodes"],
                binding._views["goal_parents"],
                binding._views["connection_indices"],
                binding._views["distance_weight"],
            ],
            outputs=[
                binding._views["output_paths"],
                binding._views["path_counts"],
                binding._views["path_lengths"],
            ],
            block_dim=self.profile.block_dim,
            stream=warp_stream,
        )
        return binding.output_paths, binding.path_counts, binding.path_lengths
