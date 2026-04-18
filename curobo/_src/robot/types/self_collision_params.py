# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.autograd.profiler as profiler

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise, log_debug


@dataclass
class SelfCollisionKinematicsCfg:
    """Dataclass that stores self collision attributes to pass to cuda kernel."""

    #: Number of spheres in the robot model.
    num_spheres: int = 0

    #: Offset radii for each sphere. This is used to inflate the spheres for self collision
    #: detection. Shape is [num_spheres].
    sphere_padding: Optional[torch.Tensor] = None

    #: Sphere index to use for a given thread. Shape is [num_collision_checks, 2] with each
    #: row having [sphere_idx_1, sphere_idx_2].
    collision_pairs: Optional[torch.Tensor] = None

    _num_checks_per_thread_large_collision_pairs: int = 256
    _max_threads_per_block_large_collision_pairs: int = 512

    _max_threads_per_block_small_collision_pairs: int = 64
    _num_checks_per_thread_small_collision_pairs: int = 32

    @property
    def num_checks_per_thread(self):
        num_collision_pairs = self.collision_pairs.shape[0]
        if num_collision_pairs > 1000:
            return self._num_checks_per_thread_large_collision_pairs
        else:
            return self._num_checks_per_thread_small_collision_pairs

    @property
    def max_threads_per_block(self):
        num_collision_pairs = self.collision_pairs.shape[0]
        if num_collision_pairs > 1000:
            return self._max_threads_per_block_large_collision_pairs
        else:
            return self._max_threads_per_block_small_collision_pairs

    @property
    def num_blocks_per_batch(self):
        num_collision_pairs = self.collision_pairs.shape[0]
        num_blocks = math.ceil(
            num_collision_pairs / (self.num_checks_per_thread * self.max_threads_per_block)
        )
        return num_blocks

    @profiler.record_function("SelfCollisionKinematicsCfg.create_from_sphere_pair_distances")
    @staticmethod
    def create_from_sphere_pair_distances(
        sphere_pair_distances: torch.Tensor,
        sphere_padding: torch.Tensor,
    ) -> SelfCollisionKinematicsCfg:
        """Create a SelfCollisionKinematicsCfg from sphere pair distances and padding.

        Args:
            sphere_pair_distances: Tensor of shape [num_spheres, num_spheres] with pairwise distances
                between spheres.
            sphere_padding: Tensor of shape [num_spheres] with padding for each sphere.
        """
        if sphere_padding.ndim != 1:
            log_and_raise(f"sphere_padding shape {sphere_padding.shape} should be (num_spheres)")
        num_spheres = sphere_padding.shape[0]
        if sphere_pair_distances.shape != (num_spheres, num_spheres):
            log_and_raise(
                f"sphere_pair_distances shape {sphere_pair_distances.shape} should be {num_spheres, num_spheres}"
            )

        device = sphere_pair_distances.device

        coll_cpu = sphere_pair_distances.cpu()
        num_spheres = coll_cpu.shape[0]
        collision_pairs = torch.zeros((num_spheres * num_spheres, 2), dtype=torch.int16)
        collision_pairs_idx = 0
        skip_count = 0
        all_val = 0
        # count number of self collisions:
        for i in range(num_spheres):
            if torch.max(coll_cpu[i]) == -torch.inf:
                log_debug("skip" + str(i))
                continue
            for j in range(i + 1, num_spheres):
                if coll_cpu[i, j] != -torch.inf:
                    ix = collision_pairs_idx
                    collision_pairs[ix, 0] = i
                    collision_pairs[ix, 1] = j
                    collision_pairs_idx += 1
                else:
                    skip_count += 1
                all_val += 1

        collision_pairs = (
            collision_pairs[:collision_pairs_idx, :].contiguous().clone().to(device=device)
        )
        num_self_collisions = collision_pairs.shape[0]
        max_threads_per_block = 512
        log_debug("Self Collision skipped %: " + str(100 * float(skip_count) / all_val))
        log_debug("Self Collision count: " + str(num_self_collisions))
        log_debug(
            "Self Collision per thread: "
            + str(math.ceil(num_self_collisions / max_threads_per_block))
        )

        return SelfCollisionKinematicsCfg(
            num_spheres=num_spheres,
            sphere_padding=sphere_padding,
            collision_pairs=collision_pairs,
        )

    @profiler.record_function(
        "SelfCollisionKinematicsCfg.compute_sphere_pair_distance_with_link_pair_ignores"
    )
    @staticmethod
    def compute_sphere_pair_distance_with_link_pair_ignores(
        collision_link_names: List[str],
        link_name_to_sphere_index: Dict[str, int],
        self_collision_link_pair_ignores: Dict[str, List[str]],
        self_collision_link_padding: Dict[str, float],
        all_link_spheres: torch.Tensor,
        link_index_to_sphere_index: torch.Tensor,
        device_cfg: DeviceCfg,
    ) -> torch.Tensor:
        """Compute pairwise distance between spheres, considering link pair ignores.

        Inputs:
        all_link_spheres: [num_spheres, 4]
        link_name_to_index: Dict[str, int]
        link_index_to_sphere_index: [num_links, num_spheres]
        link_pair_ignores: Dict[str, List[str]]
        self_collision_sphere_padding: [num_spheres]

        Outputs:
        sphere_pair_distances: [num_spheres, num_spheres]
        self_collision_sphere_padding: [num_spheres]
        """
        # check if all tensor data is on the same device:
        if all_link_spheres.device.type != device_cfg.device.type:
            log_and_raise(
                f"all_link_spheres is on device: {all_link_spheres.device} as device_cfg is on device: {device_cfg.device}"
            )
        if link_index_to_sphere_index.device.type != device_cfg.device.type:
            log_and_raise(
                f"link_index_to_sphere_index is on device: {link_index_to_sphere_index.device} as device_cfg is on device: {device_cfg.device}"
            )

        num_spheres = all_link_spheres.shape[0]
        self_collision_sphere_padding = torch.zeros(
            num_spheres, dtype=device_cfg.dtype, device=device_cfg.device
        )
        self_collision_distance = torch.full(
            (num_spheres, num_spheres),
            fill_value=-torch.inf,
            dtype=device_cfg.dtype,
            device=device_cfg.device,
        )
        # iterate through each link:
        for j_idx, j in enumerate(collision_link_names):
            ignore_links = []
            if j in self_collision_link_pair_ignores.keys():
                ignore_links = self_collision_link_pair_ignores[j]
            link1_idx = link_name_to_sphere_index[j]
            link1_spheres_idx = torch.nonzero(link_index_to_sphere_index == link1_idx)

            rad1 = all_link_spheres[link1_spheres_idx, 3]
            if j not in self_collision_link_padding.keys():
                self_collision_link_padding[j] = 0.0
            c1 = self_collision_link_padding[j]
            self_collision_sphere_padding[link1_spheres_idx] = c1
            for _, i_name in enumerate(collision_link_names):
                if i_name == j or i_name in ignore_links:
                    continue
                if i_name not in collision_link_names:
                    log_and_raise("Self Collision Link name not found in collision_link_names")
                # find index of this link name:
                if i_name not in self_collision_link_padding.keys():
                    self_collision_link_padding[i_name] = 0.0
                c2 = self_collision_link_padding[i_name]
                link2_idx = link_name_to_sphere_index[i_name]
                # update collision distance between spheres from these two links:
                link2_spheres_idx = torch.nonzero(link_index_to_sphere_index == link2_idx)
                rad2 = all_link_spheres[link2_spheres_idx, 3]
                sp1_sp2_distance = rad1 + rad2.view(1, -1) + c1 + c2
                link1_indices = link1_spheres_idx.view(-1)
                link2_indices = link2_spheres_idx.view(-1)
                idx1, idx2 = torch.meshgrid(link1_indices, link2_indices, indexing="ij")

                self_collision_distance[idx1, idx2] = sp1_sp2_distance

        self_collision_distance = self_collision_distance.to(device=device_cfg.device)
        with profiler.record_function("robot_generator/self_collision_min"):
            d_mat = self_collision_distance
            self_collision_distance = torch.minimum(d_mat, d_mat.transpose(0, 1))

        return self_collision_distance, self_collision_sphere_padding

    @profiler.record_function("SelfCollisionKinematicsCfg.create_from_link_pairs")
    @staticmethod
    def create_from_link_pairs(
        collision_link_names: List[str],
        link_name_to_sphere_index: Dict[str, int],
        self_collision_link_pair_ignores: Dict[str, List[str]],
        self_collision_link_padding: Dict[str, float],
        all_link_spheres: torch.Tensor,
        link_index_to_sphere_index: torch.Tensor,
        device_cfg: DeviceCfg,
    ) -> SelfCollisionKinematicsCfg:
        """Create a SelfCollisionKinematicsCfg from link pairs."""
        sphere_pair_distances, self_collision_sphere_padding = (
            SelfCollisionKinematicsCfg.compute_sphere_pair_distance_with_link_pair_ignores(
                collision_link_names,
                link_name_to_sphere_index,
                self_collision_link_pair_ignores,
                self_collision_link_padding,
                all_link_spheres,
                link_index_to_sphere_index,
                device_cfg,
            )
        )

        config = SelfCollisionKinematicsCfg.create_from_sphere_pair_distances(
            sphere_pair_distances, self_collision_sphere_padding
        )
        return config
