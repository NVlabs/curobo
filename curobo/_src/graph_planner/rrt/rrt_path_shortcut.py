# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-budget GPU visibility shortcutting for RRT paths.

Candidate edges are admitted only when every collision sample fits in the
fixed state budget. Untested edges remain invisible while adjacent raw-path
edges are implicit fallbacks, so budget overflow can reduce shortcut quality
but cannot invalidate the input path.
"""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Optional

# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.util.cuda_graph_util import GraphExecutor, create_graph_executor
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import get_warp_device_stream, warp_constant_suffix, warp_kernel


@dataclass(frozen=True)
class RRTPathShortcutterConfig:
    """Compile-time constants for one batched GPU shortcut profile."""

    action_dim: int
    path_capacity: int = 16
    raw_path_capacity: Optional[int] = None
    state_capacity: int = 2000
    batch_capacity: int = 1
    cspace_resolution: float = 0.02
    block_dim: int = 128

    def __post_init__(self) -> None:
        """Validate the fixed deployment shape."""
        if self.action_dim <= 0:
            log_and_raise("action_dim must be positive")
        if self.path_capacity < 3:
            log_and_raise("path_capacity must be at least three")
        if self.raw_path_capacity is None:
            object.__setattr__(self, "raw_path_capacity", self.path_capacity)
        if self.raw_path_capacity < self.path_capacity:
            log_and_raise("raw_path_capacity must cover path_capacity")
        if self.batch_capacity < 1:
            log_and_raise("batch_capacity must be positive")
        if self.state_capacity < 2 * self.batch_capacity:
            log_and_raise("state_capacity must provide at least two states per batch lane")
        if self.cspace_resolution <= 0.0:
            log_and_raise("cspace_resolution must be positive")
        if self.block_dim <= 0 or self.block_dim > 256:
            log_and_raise("block_dim must be in [1, 256]")

    @property
    def edge_capacity(self) -> int:
        """Return the number of non-adjacent ordered-DAG edges."""
        return (self.path_capacity - 1) * (self.path_capacity - 2) // 2

    @property
    def states_per_path(self) -> int:
        """Return the deterministic state-budget partition for each lane."""
        return self.state_capacity // self.batch_capacity


@dataclass(frozen=True)
class _RRTPathShortcutKernelBundle:
    setup: Any
    fill_edges: Any
    reduce_edges: Any
    select_path: Any


def _build_shortcut_pair_tables(path_capacity: int) -> tuple[list[tuple[int, int]], list[int]]:
    """Build the ordered non-adjacent edge table shared by Python and AOT export."""
    pairs = [
        (source, source + gap)
        for gap in range(path_capacity - 1, 1, -1)
        for source in range(path_capacity - gap)
    ]
    lookup = [-1] * (path_capacity * path_capacity)
    for edge, (source, destination) in enumerate(pairs):
        lookup[source * path_capacity + destination] = edge
    return pairs, lookup


@lru_cache(maxsize=None)
def _make_shortcut_kernels(
    action_dim: int,
    path_capacity: int,
    raw_path_capacity: int,
    state_capacity: int,
    cspace_resolution: float,
    edge_capacity: int,
    block_dim: int,
) -> _RRTPathShortcutKernelBundle:
    """Generate and cache one robot/shape-specialized kernel bundle."""
    dof = wp.constant(action_dim)
    max_waypoints = wp.constant(path_capacity)
    max_raw_waypoints = wp.constant(raw_path_capacity)
    max_states = wp.constant(state_capacity)
    resolution = wp.constant(cspace_resolution)
    edge_count = wp.constant(edge_capacity)
    suffix = warp_constant_suffix(
        "rrt_path_shortcut_v2",
        action_dim,
        path_capacity,
        raw_path_capacity,
        state_capacity,
        cspace_resolution,
        block_dim,
    )

    @warp_kernel(
        f"rrt_path_shortcut_setup_{suffix}",
        enable_backward=False,
    )
    def setup_kernel(
        raw_path: wp.array2d(dtype=wp.float32),
        path_count: wp.array(dtype=wp.int32),
        distance_weight: wp.array(dtype=wp.float32),
        pair_sources: wp.array(dtype=wp.int32),
        pair_destinations: wp.array(dtype=wp.int32),
        edge_offsets: wp.array(dtype=wp.int32),
        edge_steps: wp.array(dtype=wp.int32),
        edge_tested: wp.array(dtype=wp.bool),
        status: wp.array(dtype=wp.int32),
    ):
        plan = wp.tid()
        plan_count = path_count.shape[0]
        raw_offset = plan * max_raw_waypoints
        edge_offset = plan * edge_count
        state_capacity_per_plan = max_states / plan_count
        state_offset = plan * state_capacity_per_plan
        status_offset = plan * 5
        raw_count = path_count[plan]
        n = wp.clamp(raw_count, wp.int32(0), wp.int32(max_waypoints))
        used = int(0)
        tested = int(0)
        overflow = int(0)
        for edge in range(edge_count):
            source = pair_sources[edge]
            destination = pair_destinations[edge]
            flat_edge = edge_offset + edge
            edge_offsets[flat_edge] = wp.int32(-1)
            edge_steps[flat_edge] = wp.int32(0)
            edge_tested[flat_edge] = False
            if raw_count <= max_waypoints and source < n and destination < n:
                weighted_max = float(0.0)
                for joint in range(dof):
                    delta = wp.abs(
                        (
                            raw_path[raw_offset + destination, joint]
                            - raw_path[raw_offset + source, joint]
                        )
                        * distance_weight[joint]
                    )
                    if delta > weighted_max:
                        weighted_max = delta
                # Match LinearConnector's ``ceil(d / r) + 1`` complete grid,
                # but do not re-evaluate either endpoint. RRT reconstruction
                # already guarantees every raw waypoint is feasible, so only
                # the interior samples consume rollout capacity.
                required_steps = wp.max(int(wp.ceil(weighted_max / resolution)) - 1, 0)
                if required_steps <= state_capacity_per_plan - used:
                    edge_offsets[flat_edge] = wp.int32(state_offset + used)
                    edge_steps[flat_edge] = wp.int32(required_steps)
                    edge_tested[flat_edge] = True
                    used = used + required_steps
                    tested = tested + 1
                else:
                    overflow = overflow + 1
        status[status_offset + 0] = wp.int32(0)
        status[status_offset + 1] = wp.int32(used)
        status[status_offset + 2] = wp.int32(tested)
        status[status_offset + 3] = wp.int32(overflow)
        status[status_offset + 4] = wp.int32(raw_count > max_waypoints)

    @warp_kernel(
        f"rrt_path_shortcut_fill_edges_{suffix}",
        enable_backward=False,
    )
    def fill_edges_kernel(
        raw_path: wp.array2d(dtype=wp.float32),
        pair_sources: wp.array(dtype=wp.int32),
        pair_destinations: wp.array(dtype=wp.int32),
        edge_offsets: wp.array(dtype=wp.int32),
        edge_steps: wp.array(dtype=wp.int32),
        edge_tested: wp.array(dtype=wp.bool),
        packed_states: wp.array2d(dtype=wp.float32),
    ):
        flat_edge = wp.tid()
        plan = flat_edge / edge_count
        edge = flat_edge - plan * edge_count
        raw_offset = plan * max_raw_waypoints
        if edge_tested[flat_edge]:
            source = pair_sources[edge]
            destination = pair_destinations[edge]
            offset = edge_offsets[flat_edge]
            steps = edge_steps[flat_edge]
            denominator = float(steps + 1)
            for step in range(steps):
                alpha = float(step + 1) / denominator
                for joint in range(dof):
                    start_value = raw_path[raw_offset + source, joint]
                    packed_states[offset + step, joint] = start_value + alpha * (
                        raw_path[raw_offset + destination, joint] - start_value
                    )

    @warp_kernel(
        f"rrt_path_shortcut_reduce_edges_{suffix}",
        enable_backward=False,
    )
    def reduce_edges_kernel(
        feasible: wp.array(dtype=wp.bool),
        edge_offsets: wp.array(dtype=wp.int32),
        edge_steps: wp.array(dtype=wp.int32),
        edge_tested: wp.array(dtype=wp.bool),
        edge_visible: wp.array(dtype=wp.bool),
    ):
        edge = wp.tid()
        visible = bool(False)
        if edge_tested[edge]:
            visible = bool(True)
            offset = edge_offsets[edge]
            steps = edge_steps[edge]
            for step in range(steps):
                if not feasible[offset + step]:
                    visible = bool(False)
        edge_visible[edge] = visible

    @warp_kernel(
        f"rrt_path_shortcut_select_path_{suffix}",
        enable_backward=False,
    )
    def select_path_kernel(
        raw_path: wp.array2d(dtype=wp.float32),
        path_count: wp.array(dtype=wp.int32),
        raw_path_length: wp.array(dtype=wp.float32),
        distance_weight: wp.array(dtype=wp.float32),
        edge_lookup: wp.array(dtype=wp.int32),
        edge_visible: wp.array(dtype=wp.bool),
        dp_hops: wp.array(dtype=wp.int32),
        dp_cost: wp.array(dtype=wp.float32),
        parents: wp.array(dtype=wp.int32),
        output_path: wp.array2d(dtype=wp.float32),
        output_indices: wp.array(dtype=wp.int32),
        status: wp.array(dtype=wp.int32),
        path_length: wp.array(dtype=wp.float32),
        output_count: wp.array(dtype=wp.int32),
        success: wp.array(dtype=wp.bool),
    ):
        plan = wp.tid()
        raw_offset = plan * max_raw_waypoints
        work_offset = plan * max_waypoints
        edge_offset = plan * edge_count
        status_offset = plan * 5
        raw_count = path_count[plan]
        if raw_count < 2:
            status[status_offset] = wp.int32(0)
            output_count[plan] = wp.int32(0)
            success[plan] = False
            path_length[plan] = wp.float32(wp.inf)
            return
        if raw_count > max_waypoints:
            for waypoint in range(max_raw_waypoints):
                if waypoint < raw_count:
                    output_indices[raw_offset + waypoint] = wp.int32(waypoint)
                    for joint in range(dof):
                        output_path[raw_offset + waypoint, joint] = raw_path[
                            raw_offset + waypoint, joint
                        ]
            status[status_offset] = raw_count
            output_count[plan] = raw_count
            success[plan] = True
            path_length[plan] = raw_path_length[plan]
            return
        n = raw_count
        goal = n - 1
        infinity = float(3.402823466e38)
        infinite_hops = int(max_waypoints + 1)

        for waypoint in range(max_waypoints):
            dp_hops[work_offset + waypoint] = wp.int32(infinite_hops)
            dp_cost[work_offset + waypoint] = infinity
            parents[work_offset + waypoint] = wp.int32(-1)
            output_indices[raw_offset + waypoint] = goal
            for joint in range(dof):
                output_path[raw_offset + waypoint, joint] = raw_path[raw_offset + goal, joint]
        dp_hops[work_offset] = wp.int32(0)
        dp_cost[work_offset] = float(0.0)

        for destination in range(1, max_waypoints):
            if destination < n:
                best_hops = infinite_hops
                best_cost = infinity
                best_parent = int(max_waypoints)
                for source in range(max_waypoints):
                    if source < destination:
                        lookup = edge_lookup[source * max_waypoints + destination]
                        visible = destination == source + 1
                        if lookup >= 0:
                            visible = visible or edge_visible[edge_offset + lookup]
                        if visible:
                            edge_cost_squared = float(0.0)
                            for joint in range(dof):
                                delta = (
                                    raw_path[raw_offset + destination, joint]
                                    - raw_path[raw_offset + source, joint]
                                ) * distance_weight[joint]
                                edge_cost_squared = edge_cost_squared + delta * delta
                            candidate_hops = int(dp_hops[work_offset + source]) + 1
                            candidate_cost = dp_cost[work_offset + source] + wp.sqrt(
                                edge_cost_squared
                            )
                            better = candidate_hops < best_hops
                            better = better or (
                                candidate_hops == best_hops and candidate_cost < best_cost
                            )
                            better = better or (
                                candidate_hops == best_hops
                                and candidate_cost == best_cost
                                and source < best_parent
                            )
                            if better:
                                best_hops = candidate_hops
                                best_cost = candidate_cost
                                best_parent = source
                dp_hops[work_offset + destination] = wp.int32(best_hops)
                dp_cost[work_offset + destination] = best_cost
                parents[work_offset + destination] = wp.int32(best_parent)

        selected_count = int(dp_hops[work_offset + goal]) + 1
        cursor = int(goal)
        for reverse_step in range(max_waypoints):
            if reverse_step < selected_count:
                output_slot = selected_count - 1 - reverse_step
                output_indices[raw_offset + output_slot] = wp.int32(cursor)
                for joint in range(dof):
                    output_path[raw_offset + output_slot, joint] = raw_path[
                        raw_offset + cursor, joint
                    ]
                if cursor > 0:
                    cursor = int(parents[work_offset + cursor])
        status[status_offset] = wp.int32(selected_count)
        output_count[plan] = wp.int32(selected_count)
        success[plan] = selected_count >= 2
        path_length[plan] = dp_cost[work_offset + goal]

    kernels = (
        setup_kernel,
        fill_edges_kernel,
        reduce_edges_kernel,
        select_path_kernel,
    )
    for kernel in kernels:
        kernel.module.options.update(
            {
                "enable_backward": False,
                "block_dim": block_dim,
                "fuse_fp": False,
            }
        )
    return _RRTPathShortcutKernelBundle(*kernels)


class RRTPathShortcutter:
    """Run one fixed-budget batched shortcut pass on the current CUDA stream."""

    def __init__(
        self,
        profile: RRTPathShortcutterConfig,
        device: torch.device,
        distance_weight: torch.Tensor,
        *,
        raw_path: Optional[torch.Tensor] = None,
        path_count: Optional[torch.Tensor] = None,
        raw_path_length: Optional[torch.Tensor] = None,
    ) -> None:
        """Allocate or bind persistent Torch buffers and zero-copy Warp views."""
        device = torch.device(device)
        if device.type != "cuda":
            log_and_raise("RRTPathShortcutter requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if distance_weight.dtype != torch.float32 or distance_weight.device != device:
            log_and_raise("distance_weight must be CUDA float32 on the shortcut device")
        if distance_weight.shape != (profile.action_dim,):
            log_and_raise("distance_weight shape must match action_dim")
        self.profile = profile
        self.device = device
        self.distance_weight = distance_weight
        self._kernels = _make_shortcut_kernels(
            profile.action_dim,
            profile.path_capacity,
            int(profile.raw_path_capacity),
            profile.state_capacity,
            profile.cspace_resolution,
            profile.edge_capacity,
            profile.block_dim,
        )

        pairs, lookup = _build_shortcut_pair_tables(profile.path_capacity)
        if len(pairs) != profile.edge_capacity:
            log_and_raise("shortcut pair table has an invalid size")

        batch_capacity = profile.batch_capacity
        raw_path_capacity = int(profile.raw_path_capacity)
        expected_raw_shape = (batch_capacity, raw_path_capacity, profile.action_dim)
        self.raw_path = (
            torch.empty(expected_raw_shape, device=device, dtype=torch.float32)
            if raw_path is None
            else raw_path
        )
        if (
            self.raw_path.shape != expected_raw_shape
            or self.raw_path.dtype != torch.float32
            or self.raw_path.device != device
            or not self.raw_path.is_contiguous()
        ):
            log_and_raise("raw_path must match the batched shortcut profile")
        self.path_count = (
            torch.zeros(batch_capacity, device=device, dtype=torch.int32)
            if path_count is None
            else path_count
        )
        if (
            self.path_count.shape != (batch_capacity,)
            or self.path_count.dtype != torch.int32
            or self.path_count.device != device
            or not self.path_count.is_contiguous()
        ):
            log_and_raise("path_count must match the batched shortcut profile")
        self.raw_path_length = (
            torch.full(
                (batch_capacity,),
                torch.inf,
                device=device,
                dtype=torch.float32,
            )
            if raw_path_length is None
            else raw_path_length
        )
        if (
            self.raw_path_length.shape != (batch_capacity,)
            or self.raw_path_length.dtype != torch.float32
            or self.raw_path_length.device != device
            or not self.raw_path_length.is_contiguous()
        ):
            log_and_raise("raw_path_length must match the batched shortcut profile")
        self.pair_sources = torch.tensor(
            [source for source, _ in pairs], device=device, dtype=torch.int32
        )
        self.pair_destinations = torch.tensor(
            [destination for _, destination in pairs], device=device, dtype=torch.int32
        )
        self.edge_lookup = torch.tensor(lookup, device=device, dtype=torch.int32)
        edge_shape = (batch_capacity, profile.edge_capacity)
        self.edge_offsets = torch.empty(edge_shape, device=device, dtype=torch.int32)
        self.edge_steps = torch.empty_like(self.edge_offsets)
        self.edge_tested = torch.empty(edge_shape, device=device, dtype=torch.bool)
        self.edge_visible = torch.empty_like(self.edge_tested)
        # Unused rows are never reduced, but the fixed-size rollout still
        # evaluates them. Initialize once to finite values and overwrite only
        # the packed active prefix on later calls.
        self.packed_states = torch.zeros(
            (profile.state_capacity, profile.action_dim),
            device=device,
            dtype=torch.float32,
        )
        self.feasible = torch.empty(profile.state_capacity, device=device, dtype=torch.bool)
        self.env_query_idx = torch.empty(
            profile.state_capacity,
            device=device,
            dtype=torch.int32,
        )
        self.goal_query_idx = torch.empty_like(self.env_query_idx)
        self.state_plan_idx = (
            torch.arange(
                profile.state_capacity,
                device=device,
                dtype=torch.int64,
            )
            .div_(profile.states_per_path, rounding_mode="floor")
            .clamp_max_(batch_capacity - 1)
        )
        work_shape = (batch_capacity, profile.path_capacity)
        self.dp_hops = torch.empty(work_shape, device=device, dtype=torch.int32)
        self.dp_cost = torch.empty(work_shape, device=device, dtype=torch.float32)
        self.path_length = torch.empty(batch_capacity, device=device, dtype=torch.float32)
        self.parents = torch.empty(work_shape, device=device, dtype=torch.int32)
        self.output_path = torch.empty_like(self.raw_path)
        self.output_indices = torch.empty(
            (batch_capacity, raw_path_capacity),
            device=device,
            dtype=torch.int32,
        )
        self.output_count = torch.zeros(batch_capacity, device=device, dtype=torch.int32)
        self.success = torch.zeros(batch_capacity, device=device, dtype=torch.bool)
        # [selected waypoints, used states, tested edges, overflow edges, capacity miss]
        self.status = torch.zeros((batch_capacity, 5), device=device, dtype=torch.int32)

        tensors = {
            "raw_path": self.raw_path.view(-1, profile.action_dim),
            "path_count": self.path_count,
            "raw_path_length": self.raw_path_length,
            "distance_weight": self.distance_weight,
            "pair_sources": self.pair_sources,
            "pair_destinations": self.pair_destinations,
            "edge_lookup": self.edge_lookup,
            "edge_offsets": self.edge_offsets.view(-1),
            "edge_steps": self.edge_steps.view(-1),
            "edge_tested": self.edge_tested.view(-1),
            "edge_visible": self.edge_visible.view(-1),
            "packed_states": self.packed_states,
            "feasible": self.feasible,
            "dp_hops": self.dp_hops.view(-1),
            "dp_cost": self.dp_cost.view(-1),
            "parents": self.parents.view(-1),
            "output_path": self.output_path.view(-1, profile.action_dim),
            "output_indices": self.output_indices.view(-1),
            "status": self.status.view(-1),
            "path_length": self.path_length,
            "output_count": self.output_count,
            "success": self.success,
        }
        wp.init()
        self._views = {name: wp.from_torch(tensor) for name, tensor in tensors.items()}
        loaded_modules = set()
        for kernel in (
            self._kernels.setup,
            self._kernels.fill_edges,
            self._kernels.reduce_edges,
            self._kernels.select_path,
        ):
            module_id = id(kernel.module)
            if module_id in loaded_modules:
                continue
            wp.load_module(
                module=kernel.module,
                device=str(self.device),
                block_dim=profile.block_dim,
            )
            loaded_modules.add(module_id)
        self._prepare_executor: GraphExecutor = create_graph_executor(
            capture_fn=self._launch_prepare,
            device=self.device,
            use_cuda_graph=True,
            clone_outputs=False,
            clone_inputs=False,
        )
        self._select_executor: GraphExecutor = create_graph_executor(
            capture_fn=self._launch_select,
            device=self.device,
            use_cuda_graph=True,
            clone_outputs=False,
            clone_inputs=False,
        )

    def _stream(self) -> wp.Stream:
        _, stream = get_warp_device_stream(self.raw_path)
        if stream is None:
            log_and_raise("RRT shortcut requires a CUDA Warp stream")
        return stream

    def _launch_prepare(self) -> torch.Tensor:
        """Launch fixed-address setup and candidate-state generation kernels."""
        stream = self._stream()
        wp.launch(
            self._kernels.setup,
            dim=self.profile.batch_capacity,
            inputs=[
                self._views["raw_path"],
                self._views["path_count"],
                self._views["distance_weight"],
                self._views["pair_sources"],
                self._views["pair_destinations"],
            ],
            outputs=[
                self._views["edge_offsets"],
                self._views["edge_steps"],
                self._views["edge_tested"],
                self._views["status"],
            ],
            block_dim=self.profile.block_dim,
            stream=stream,
        )
        wp.launch(
            self._kernels.fill_edges,
            dim=self.profile.batch_capacity * self.profile.edge_capacity,
            inputs=[
                self._views["raw_path"],
                self._views["pair_sources"],
                self._views["pair_destinations"],
                self._views["edge_offsets"],
                self._views["edge_steps"],
                self._views["edge_tested"],
            ],
            outputs=[self._views["packed_states"]],
            block_dim=self.profile.block_dim,
            stream=stream,
        )
        return self.packed_states

    def _prepare_batched_states(
        self,
        paths: torch.Tensor,
        path_counts: torch.Tensor,
        raw_path_lengths: torch.Tensor,
    ) -> None:
        """Stage batched paths and replay fixed candidate-state generation."""
        if paths.shape != self.raw_path.shape or paths.dtype != torch.float32:
            log_and_raise("paths must match the fixed batched raw-path buffer")
        if paths.device != self.device or not paths.is_contiguous():
            log_and_raise("paths must be contiguous on the shortcut device")
        if (
            path_counts.shape != self.path_count.shape
            or path_counts.dtype != torch.int32
            or path_counts.device != self.device
        ):
            log_and_raise("path_counts must match the fixed batched count buffer")
        if (
            raw_path_lengths.shape != self.raw_path_length.shape
            or raw_path_lengths.dtype != torch.float32
            or raw_path_lengths.device != self.device
        ):
            log_and_raise("raw_path_lengths must match the fixed batched length buffer")
        if paths.data_ptr() != self.raw_path.data_ptr():
            self.raw_path.copy_(paths)
        if path_counts.data_ptr() != self.path_count.data_ptr():
            self.path_count.copy_(path_counts)
        if raw_path_lengths.data_ptr() != self.raw_path_length.data_ptr():
            self.raw_path_length.copy_(raw_path_lengths)
        if self._prepare_executor.is_initialized:
            self._prepare_executor.replay(clone_outputs=False)
        else:
            self._prepare_executor.warmup()

    def _prepare_states(self, path: torch.Tensor) -> None:
        """Stage one path for the compatibility single-lane API."""
        if self.profile.batch_capacity != 1:
            log_and_raise("single-path shortcut requires batch_capacity=1")
        waypoint_count = path.shape[0]
        if not 2 < waypoint_count <= int(self.profile.raw_path_capacity):
            log_and_raise("path must fit the shortcut profile and contain at least 3 points")
        self.raw_path.zero_()
        self.raw_path[0, :waypoint_count].copy_(path)
        self.path_count.fill_(waypoint_count)
        path_delta = (path[1:] - path[:-1]) * self.distance_weight.view(1, -1)
        self.raw_path_length[0].copy_(torch.linalg.vector_norm(path_delta, dim=1).sum())
        if self._prepare_executor.is_initialized:
            self._prepare_executor.replay(clone_outputs=False)
        else:
            self._prepare_executor.warmup()

    def _launch_select(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Launch fixed-address feasibility reduction and DAG selection kernels."""
        stream = self._stream()
        wp.launch(
            self._kernels.reduce_edges,
            dim=self.profile.batch_capacity * self.profile.edge_capacity,
            inputs=[
                self._views["feasible"],
                self._views["edge_offsets"],
                self._views["edge_steps"],
                self._views["edge_tested"],
            ],
            outputs=[self._views["edge_visible"]],
            block_dim=self.profile.block_dim,
            stream=stream,
        )
        wp.launch(
            self._kernels.select_path,
            dim=self.profile.batch_capacity,
            inputs=[
                self._views["raw_path"],
                self._views["path_count"],
                self._views["raw_path_length"],
                self._views["distance_weight"],
                self._views["edge_lookup"],
                self._views["edge_visible"],
            ],
            outputs=[
                self._views["dp_hops"],
                self._views["dp_cost"],
                self._views["parents"],
                self._views["output_path"],
                self._views["output_indices"],
                self._views["status"],
                self._views["path_length"],
                self._views["output_count"],
                self._views["success"],
            ],
            block_dim=self.profile.block_dim,
            stream=stream,
        )
        return self.output_path, self.status

    def _select(self, clone_output: bool = True) -> tuple[torch.Tensor, dict[str, int]]:
        """Replay feasibility reduction and select one minimum-hop visible path."""
        if self._select_executor.is_initialized:
            self._select_executor.replay(clone_outputs=False)
        else:
            self._select_executor.warmup()
        if self.profile.batch_capacity != 1:
            log_and_raise("single-path selection requires batch_capacity=1")
        selected_count, used_states, tested_edges, overflow_edges, capacity_miss = (
            self.status[0].cpu().tolist()
        )
        stats = {
            "selected_waypoints": int(selected_count),
            "used_states": int(used_states),
            "tested_edges": int(tested_edges),
            "overflow_edges": int(overflow_edges),
        }
        if capacity_miss:
            stats["capacity_miss"] = int(capacity_miss)
        output = self.output_path[0, :selected_count]
        return output.clone() if clone_output else output, stats

    def shortcut_batched(
        self,
        paths: torch.Tensor,
        path_counts: torch.Tensor,
        raw_path_lengths: torch.Tensor,
        check_feasibility: Callable[..., torch.Tensor],
        env_query_idx: Optional[torch.Tensor] = None,
        goal_query_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shortcut every fixed lane with one rollout and no host synchronization."""
        self._prepare_batched_states(paths, path_counts, raw_path_lengths)
        route_kwargs = self.route_query_indices(env_query_idx, goal_query_idx)
        feasible = check_feasibility(self.packed_states, clone_output=False, **route_kwargs)
        self.feasible.copy_(feasible.reshape(-1))
        if self._select_executor.is_initialized:
            self._select_executor.replay(clone_outputs=False)
        else:
            self._select_executor.warmup()
        return (
            self.output_path,
            self.output_count,
            self.path_length,
            self.success,
            self.status,
        )

    def route_query_indices(
        self,
        env_query_idx: Optional[torch.Tensor],
        goal_query_idx: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Expand per-lane routing indices into fixed per-state buffers."""
        route_kwargs = {}
        for name, query_idx, destination in (
            ("env_query_idx", env_query_idx, self.env_query_idx),
            ("goal_query_idx", goal_query_idx, self.goal_query_idx),
        ):
            if query_idx is None:
                continue
            if (
                query_idx.shape != (self.profile.batch_capacity,)
                or query_idx.dtype != torch.int32
                or query_idx.device != self.device
            ):
                log_and_raise(f"{name} must have one CUDA int32 value per batch lane")
            torch.index_select(query_idx, 0, self.state_plan_idx, out=destination)
            route_kwargs[name] = destination
        return route_kwargs

    def reset_cuda_graph(self) -> None:
        """Release both fixed-stage CUDA graphs without reallocating buffers."""
        self._prepare_executor.reset()
        self._select_executor.reset()

    def shortcut(
        self,
        path: torch.Tensor,
        check_feasibility: Callable[..., torch.Tensor],
        env_query_idx: Optional[torch.Tensor] = None,
        goal_query_idx: Optional[torch.Tensor] = None,
        clone_output: bool = True,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        """Generate, validate, and select one shortcut without a host round loop."""
        if (
            path.dtype != torch.float32
            or path.device != self.device
            or path.ndim != 2
            or path.shape[1] != self.profile.action_dim
        ):
            log_and_raise("path must be CUDA float32 with shape (waypoints, action_dim)")
        self._prepare_states(path)
        if env_query_idx is None and goal_query_idx is None:
            feasible = check_feasibility(self.packed_states, clone_output=False)
        else:
            expanded_env = None
            if env_query_idx is not None:
                if (
                    env_query_idx.numel() != 1
                    or env_query_idx.dtype != torch.int32
                    or env_query_idx.device != self.device
                ):
                    log_and_raise(
                        "env_query_idx must be one CUDA int32 value on the shortcut device"
                    )
                self.env_query_idx.copy_(
                    env_query_idx.reshape(1).expand(self.profile.state_capacity)
                )
                expanded_env = self.env_query_idx
            expanded_goal = None
            if goal_query_idx is not None:
                if (
                    goal_query_idx.numel() != 1
                    or goal_query_idx.dtype != torch.int32
                    or goal_query_idx.device != self.device
                ):
                    log_and_raise(
                        "goal_query_idx must be one CUDA int32 value on the shortcut device"
                    )
                self.goal_query_idx.copy_(
                    goal_query_idx.reshape(1).expand(self.profile.state_capacity)
                )
                expanded_goal = self.goal_query_idx
            route_kwargs = {"clone_output": False}
            if expanded_env is not None:
                route_kwargs["env_query_idx"] = expanded_env
            if expanded_goal is not None:
                route_kwargs["goal_query_idx"] = expanded_goal
            feasible = check_feasibility(self.packed_states, **route_kwargs)
        self.feasible.copy_(feasible.reshape(-1))
        return self._select(clone_output=clone_output)


__all__ = ["RRTPathShortcutterConfig", "RRTPathShortcutter"]
