# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for reducing degrees of freedom in KinematicsParams.

This module provides functionality to reduce DOF by removing joints that are only needed
for collision checking, effectively creating a smaller kinematic structure that only includes
joints required for pose computation of desired links.
"""

from __future__ import annotations

# Standard Library
from typing import List, Optional

# Third Party
import torch

# CuRobo
from curobo._src.robot.types.cspace_params import CSpaceParams
from curobo._src.robot.types.joint_limits import JointLimits
from curobo._src.robot.types.kinematics_params import KinematicsParams
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_ops import append_joints_to_state
from curobo._src.util.logging import log_info, log_warn


class KinematicsReducer:
    """Utility class for reducing degrees of freedom in kinematic configurations.

    This class provides methods to analyze kinematic structures and create reduced DOF
    configurations that exclude collision-only joints while preserving pose computation
    capabilities for desired links.
    """

    @classmethod
    def reduce_dof(
        cls,
        original_config: KinematicsParams,
        desired_link_names: List[str],
        remove_collision_spheres: bool = True,
    ) -> KinematicsParams:
        """Reduce DOF by keeping only joints needed for specified links.

        This method analyzes the kinematic structure to identify which joints are actually
        needed to compute poses for the desired links, then creates a new KinematicsParams
        with reduced DOF that excludes collision-only joints. All removed joints are automatically
        added to the lock_jointstate for reconstruction after optimization.

        Args:
            original_config: Original kinematics configuration
            desired_link_names: Names of links to keep poses for
            remove_collision_spheres: Whether to remove collision spheres (should be True for DOF reduction)

        Returns:
            KinematicsParams: New configuration with reduced DOF. The lock_jointstate will
                contain all joints that were removed during reduction, set to their default joint
                positions.

        Raises:
            ValueError: If none of the desired links are found in the original configuration

        Example:
            >>> # Reduce robot to only end effector joints
            >>> reduced_config = KinematicsReducer.reduce_dof(robot_config, ["end_effector"])
            >>> print(f"Reduced from {robot_config.num_dof} to {reduced_config.num_dof} DOF")
            >>>
            >>> # After optimization, reconstruct full joint state
            >>> optimized_reduced_state = optimize_trajectory(reduced_config)
            >>> full_joint_state = KinematicsReducer.reconstruct_joint_state(
            ...     optimized_reduced_state, reduced_config.lock_jointstate
            ... )
        """
        #return original_config
        #log_and_raise("Not implemented yet")
        if not remove_collision_spheres:
            log_warn("remove_collision_spheres=False may not provide full DOF reduction benefits")

        # Identify the structure we need to keep
        needed_link_indices, needed_joints, keep_stored_indices = cls._identify_needed_structure(
            original_config, desired_link_names
        )

        log_info(f"Reducing DOF from {original_config.num_dof} to {len(needed_joints)} joints")
        log_info(
            f"Reducing links from {len(original_config.tool_frames)} to {len(desired_link_names)} stored links"
        )
        log_info(f"Keeping {len(needed_link_indices)} links in kinematic structure")

        # Rebuild all kinematic tensors with the reduced structure
        reduced_config = cls._rebuild_kinematic_tensors(
            original_config,
            needed_link_indices,
            needed_joints,
            keep_stored_indices,
            desired_link_names,
        )

        # Make tensors contiguous for optimal performance
        reduced_config.make_contiguous()

        return reduced_config

    @classmethod
    def reconstruct_joint_state(
        cls,
        reduced_joint_state: JointState,
        lock_jointstate: Optional[JointState],
        target_joint_names: Optional[List[str]] = None,
    ) -> JointState:
        """Reconstruct full joint state from reduced optimization result and locked joints.

        This method combines the optimized reduced joint state with the locked joints
        to create a complete joint state that can be used with the original robot model.

        Args:
            reduced_joint_state: Joint state from reduced DOF optimization
            lock_jointstate: Locked joints from reduced kinematics config
            target_joint_names: Optional list of joint names for output ordering.
                If None, uses reduced + locked joint names.

        Returns:
            JointState: Complete joint state with all joints

        Example:
            >>> # After optimization with reduced config
            >>> optimized_reduced_state = optimize_trajectory(reduced_config)
            >>>
            >>> # Reconstruct full joint state
            >>> full_joint_state = KinematicsReducer.reconstruct_joint_state(
            ...     optimized_reduced_state, reduced_config.lock_jointstate
            ... )
            >>>
            >>> # Use with original robot model
            >>> full_robot_state = original_robot_model.compute_kinematics(full_joint_state)
        """
        if lock_jointstate is None:
            if target_joint_names is not None:
                return reduced_joint_state.reorder(target_joint_names)
            return reduced_joint_state

        # Ensure both joint states are on the same device before concatenating
        # Move lock_jointstate to the device of reduced_joint_state
        if lock_jointstate.device != reduced_joint_state.device:
            lock_jointstate = lock_jointstate.to(reduced_joint_state.device_cfg)

        # Combine reduced and locked joint states
        full_joint_state = append_joints_to_state(reduced_joint_state, lock_jointstate)

        # Reorder to target joint names if specified
        if target_joint_names is not None:
            full_joint_state = full_joint_state.reorder(target_joint_names)

        return full_joint_state

    @staticmethod
    def _create_joint_limits_subset(
        original_limits: JointLimits, joint_names: List[str]
    ) -> JointLimits:
        """Create a subset of JointLimits for specified joints.

        Args:
            original_limits: Original joint limits
            joint_names: Names of joints to keep

        Returns:
            JointLimits: Subset of joint limits
        """
        # Find indices of joints to keep
        joint_indices = [original_limits.joint_names.index(name) for name in joint_names]
        joint_indices_tensor = torch.tensor(joint_indices, dtype=torch.long)

        # Extract subset of limits
        position = original_limits.position[:, joint_indices_tensor]
        velocity = original_limits.velocity[:, joint_indices_tensor]
        acceleration = original_limits.acceleration[:, joint_indices_tensor]
        jerk = original_limits.jerk[:, joint_indices_tensor]
        effort = None
        if original_limits.effort is not None:
            effort = original_limits.effort[:, joint_indices_tensor]

        return JointLimits(
            joint_names=joint_names,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            effort=effort,
            device_cfg=original_limits.device_cfg,
        )

    @staticmethod
    def _create_cspace_subset(
        original_cspace: CSpaceParams, joint_names: List[str]
    ) -> CSpaceParams:
        """Create a subset of CSpaceParams for specified joints.

        Args:
            original_cspace: Original CSpace configuration
            joint_names: Names of joints to keep

        Returns:
            CSpaceParams: Subset of CSpace configuration
        """
        # Find indices of joints to keep
        joint_indices = [original_cspace.joint_names.index(name) for name in joint_names]
        joint_indices_tensor = torch.tensor(
            joint_indices, dtype=torch.long, device=original_cspace.device_cfg.device
        )

        # Extract subset of cspace parameters
        default_joint_position = None
        if original_cspace.default_joint_position is not None:
            default_joint_position = original_cspace.default_joint_position[joint_indices_tensor]

        cspace_distance_weight = None
        if original_cspace.cspace_distance_weight is not None:
            cspace_distance_weight = original_cspace.cspace_distance_weight[joint_indices_tensor]

        null_space_weight = None
        if original_cspace.null_space_weight is not None:
            null_space_weight = original_cspace.null_space_weight[joint_indices_tensor]

        null_space_maximum_distance = None
        if original_cspace.null_space_maximum_distance is not None:
            null_space_maximum_distance = original_cspace.null_space_maximum_distance[
                joint_indices_tensor
            ]

        max_acceleration = original_cspace.max_acceleration[joint_indices_tensor]
        max_jerk = original_cspace.max_jerk[joint_indices_tensor]
        velocity_scale = original_cspace.velocity_scale[joint_indices_tensor]
        acceleration_scale = original_cspace.acceleration_scale[joint_indices_tensor]
        jerk_scale = original_cspace.jerk_scale[joint_indices_tensor]

        position_limit_clip = original_cspace.position_limit_clip
        if isinstance(position_limit_clip, torch.Tensor):
            position_limit_clip = position_limit_clip[joint_indices_tensor]

        return CSpaceParams(
            joint_names=joint_names,
            default_joint_position=default_joint_position,
            cspace_distance_weight=cspace_distance_weight,
            null_space_weight=null_space_weight,
            null_space_maximum_distance=null_space_maximum_distance,
            device_cfg=original_cspace.device_cfg,
            max_acceleration=max_acceleration,
            max_jerk=max_jerk,
            velocity_scale=velocity_scale,
            acceleration_scale=acceleration_scale,
            jerk_scale=jerk_scale,
            position_limit_clip=position_limit_clip,
        )



    @staticmethod
    def _identify_needed_structure(
        config: KinematicsParams, desired_link_names: List[str]
    ) -> tuple[List[int], List[int], List[int]]:
        """Identify which links and joints are needed for the desired stored links.

        Args:
            config: Original kinematics configuration
            desired_link_names: Links to keep poses for

        Returns:
            Tuple containing:
                - needed_link_indices: Global link indices needed
                - needed_joints: Joint indices needed
                - keep_stored_indices: Indices in tool_frame_map to keep
        """
        # Step 1: Find which stored links we want to keep
        keep_stored_indices = []
        for link_name in desired_link_names:
            if link_name in config.tool_frames:
                keep_stored_indices.append(config.tool_frames.index(link_name))

        if len(keep_stored_indices) == 0:
            raise ValueError(
                f"None of the desired links {desired_link_names} found in {config.tool_frames}"
            )

        # Step 2: Extract all link indices needed for these stored links using kinematic chains
        needed_link_indices = set()
        for stored_idx in keep_stored_indices:
            # First find the global link index for this stored link
            stored_link_name = config.tool_frames[stored_idx]
            if config.link_name_to_idx_map is None:
                raise ValueError("link_name_to_idx_map is required for DOF reduction")

            global_link_idx = config.link_name_to_idx_map[stored_link_name]

            # Find this link's chain data by looking for its global index in the chain structure
            # The chain data is built for ALL links, indexed by their position in the full kinematic tree
            start = config.link_chain_offsets[global_link_idx].item()
            end = config.link_chain_offsets[global_link_idx + 1].item()
            chain_indices = config.link_chain_data[start:end]
            needed_link_indices.update(chain_indices.tolist())

        needed_link_indices = sorted(list(needed_link_indices))

        # Step 3: Find needed joints by looking at joint_map for needed links
        needed_joints = set()
        for link_idx in needed_link_indices:
            joint_idx = config.joint_map[link_idx].item()
            if joint_idx >= 0:  # -1 means no joint (base link)
                needed_joints.add(joint_idx)

        needed_joints = sorted(list(needed_joints))

        return needed_link_indices, needed_joints, keep_stored_indices

    @staticmethod
    def _rebuild_kinematic_tensors(
        config: KinematicsParams,
        needed_link_indices: List[int],
        needed_joints: List[int],
        keep_stored_indices: List[int],
        desired_link_names: List[str],
    ) -> KinematicsParams:
        """Rebuild all kinematic tensors with reduced structure.

        Args:
            config: Original configuration
            needed_link_indices: Link indices to keep
            needed_joints: Joint indices to keep
            keep_stored_indices: Stored link indices to keep
            desired_link_names: Names of links to store poses for

        Returns:
            KinematicsParams: New configuration with reduced DOF
        """
        device = config.fixed_transforms.device
        dtype = config.fixed_transforms.dtype

        # Create mapping from old indices to new indices
        old_to_new_link = {old_idx: new_idx for new_idx, old_idx in enumerate(needed_link_indices)}
        old_to_new_joint = {old_idx: new_idx for new_idx, old_idx in enumerate(needed_joints)}

        # 1. Rebuild fixed_transforms (subset of original)
        new_fixed_transforms = config.fixed_transforms[needed_link_indices]

        # 2. Rebuild link_map (parent relationships, reindexed)
        new_link_map = torch.zeros(len(needed_link_indices), dtype=torch.int16, device=device)
        for new_idx, old_idx in enumerate(needed_link_indices):
            old_parent = config.link_map[old_idx].item()
            if old_parent in old_to_new_link:
                new_link_map[new_idx] = old_to_new_link[old_parent]
            else:
                new_link_map[new_idx] = 0  # Should be base link

        # 3. Rebuild joint_map (reindexed joint assignments)
        new_joint_map = torch.full(
            (len(needed_link_indices),), -1, dtype=torch.int16, device=device
        )
        for new_idx, old_idx in enumerate(needed_link_indices):
            old_joint = config.joint_map[old_idx].item()
            if old_joint >= 0 and old_joint in old_to_new_joint:
                new_joint_map[new_idx] = old_to_new_joint[old_joint]

        # 4. Rebuild joint_map_type
        new_joint_map_type = torch.zeros(len(needed_link_indices), dtype=torch.int8, device=device)
        for new_idx, old_idx in enumerate(needed_link_indices):
            new_joint_map_type[new_idx] = config.joint_map_type[old_idx]

        # 5. Rebuild tool_frame_map (which reduced links to store poses for)
        new_tool_frame_map = torch.zeros(len(desired_link_names), dtype=torch.int16, device=device)
        for new_store_idx, old_stored_idx in enumerate(keep_stored_indices):
            # Find the global link index this stored link referred to
            old_global_link_idx = config.tool_frame_map[old_stored_idx].item()
            # Map to new position in reduced kinematic tree
            new_tool_frame_map[new_store_idx] = old_to_new_link[old_global_link_idx]

        # 6. Rebuild joint_offset_map
        # joint_offset_map is indexed by link and has 2 values per link
        new_joint_offset_data = []
        for old_idx in needed_link_indices:
            old_start = old_idx * 2
            new_joint_offset_data.extend(
                config.joint_offset_map[old_start : old_start + 2].tolist()
            )
        new_joint_offset_map = torch.tensor(
            new_joint_offset_data, device=device, dtype=torch.float32
        )

        # 7. Rebuild kinematic chains for all links in reduced structure (reindexed).
        # The kernel indexes linkChainOffsets by global link index
        # (toolFrameMap[i] gives a link index, then linkChainOffsets[link_idx]),
        # so the CSR must have num_links + 1 entries.
        new_chain_data = []
        new_chain_offsets = [0]

        for old_idx in needed_link_indices:
            start = config.link_chain_offsets[old_idx].item()
            end = config.link_chain_offsets[old_idx + 1].item()
            old_chain = config.link_chain_data[start:end]

            new_chain = [
                old_to_new_link[idx.item()] for idx in old_chain if idx.item() in old_to_new_link
            ]
            new_chain_data.extend(new_chain)
            new_chain_offsets.append(len(new_chain_data))

        new_link_chain_data = torch.tensor(new_chain_data, dtype=torch.int16, device=device)
        new_link_chain_offsets = torch.tensor(new_chain_offsets, dtype=torch.int16, device=device)

        # 8. Rebuild joint-link relationships (reindexed)
        new_joint_links_data = []
        new_joint_links_offsets = [0]

        for new_joint_idx, old_joint_idx in enumerate(needed_joints):
            start = config.joint_links_offsets[old_joint_idx].item()
            end = config.joint_links_offsets[old_joint_idx + 1].item()
            old_links = config.joint_links_data[start:end]

            # Reindex to new link indices
            new_links = [
                old_to_new_link[idx.item()] for idx in old_links if idx.item() in old_to_new_link
            ]
            new_joint_links_data.extend(new_links)
            new_joint_links_offsets.append(len(new_joint_links_data))

        new_joint_links_data_tensor = torch.tensor(
            new_joint_links_data, dtype=torch.int16, device=device
        )
        new_joint_links_offsets_tensor = torch.tensor(
            new_joint_links_offsets, dtype=torch.int16, device=device
        )

        # 9. Rebuild joint_affects_endeffector matrix
        new_joint_affects_ee = torch.zeros(
            (len(needed_joints) * len(desired_link_names)), dtype=torch.bool, device=device
        )

        for new_joint_idx, old_joint_idx in enumerate(needed_joints):
            for new_ee_idx, old_stored_idx in enumerate(keep_stored_indices):
                old_flat_idx = old_joint_idx * len(config.tool_frames) + old_stored_idx
                new_flat_idx = new_joint_idx * len(desired_link_names) + new_ee_idx
                new_joint_affects_ee[new_flat_idx] = config.joint_affects_endeffector[old_flat_idx]

        # 10. Extract other needed data
        new_joint_names = [config.joint_names[i] for i in needed_joints]

        # 11. Build reduced joint limits and cspace
        new_joint_limits = KinematicsReducer._create_joint_limits_subset(
            config.joint_limits, new_joint_names
        )
        new_cspace = KinematicsReducer._create_cspace_subset(config.cspace, new_joint_names)

        # 12. Handle lock_jointstate - add all removed joints to lock state
        new_lock_jointstate = None

        # Find joints that were removed during DOF reduction
        removed_joint_names = [name for name in config.joint_names if name not in new_joint_names]

        # Start with existing lock joints that are not in the reduced structure
        existing_lock_joints = []
        existing_lock_positions = []
        if config.lock_jointstate is not None:
            # Work with unique joint names to avoid duplicates
            unique_lock_joints = []
            unique_lock_positions = []
            seen_joints = set()

            # First, extract unique joints and their positions
            for i, joint_name in enumerate(config.lock_jointstate.joint_names):
                if joint_name not in seen_joints:
                    seen_joints.add(joint_name)
                    unique_lock_joints.append(joint_name)

                    # Handle different position tensor dimensions
                    if config.lock_jointstate.position.ndim == 1:
                        pos_value = config.lock_jointstate.position[
                            len(unique_lock_joints) - 1
                        ].item()
                    else:
                        pos_value = config.lock_jointstate.position[
                            0, len(unique_lock_joints) - 1
                        ].item()
                    unique_lock_positions.append(pos_value)

            # Now filter to only include joints not in the reduced structure
            for i, joint_name in enumerate(unique_lock_joints):
                if joint_name not in new_joint_names:
                    existing_lock_joints.append(joint_name)
                    existing_lock_positions.append(unique_lock_positions[i])

        # Add removed joints that are not already in the existing lock joints
        final_lock_joint_names = existing_lock_joints.copy()
        final_lock_positions = existing_lock_positions.copy()

        for joint_name in removed_joint_names:
            if joint_name not in final_lock_joint_names:  # Avoid duplicates
                joint_idx = config.joint_names.index(joint_name)
                default_joint_position = config.cspace.default_joint_position[joint_idx].item()
                final_lock_joint_names.append(joint_name)
                final_lock_positions.append(default_joint_position)

        # Create final lock joint state if we have any locked joints
        if len(final_lock_joint_names) > 0:
            new_lock_jointstate = JointState.from_position(
                torch.tensor(final_lock_positions, device=config.cspace.default_joint_position.device),
                joint_names=final_lock_joint_names,
            )

        # 13. Update link_name_to_idx_map
        new_link_name_to_idx_map = None
        if config.link_name_to_idx_map is not None:
            new_link_name_to_idx_map = {}
            for link_name, old_idx in config.link_name_to_idx_map.items():
                if old_idx in old_to_new_link:
                    new_link_name_to_idx_map[link_name] = old_to_new_link[old_idx]

        # 14. Handle mimic joints
        new_mimic_joints = None
        if config.mimic_joints is not None:
            new_mimic_joints = {}
            for joint_name, mimic_data in config.mimic_joints.items():
                if joint_name in new_joint_names:
                    new_mimic_joints[joint_name] = mimic_data

        # 15. Handle link_masses_com and link_inertias
        new_link_masses_com = None
        if config.link_masses_com is not None:
            new_link_masses_com = config.link_masses_com[needed_link_indices]

        new_link_inertias = None
        if config.link_inertias is not None:
            new_link_inertias = config.link_inertias[needed_link_indices]

        # Create new configuration with reduced structure
        return KinematicsParams(
            fixed_transforms=new_fixed_transforms,
            link_map=new_link_map,
            joint_map=new_joint_map,
            joint_map_type=new_joint_map_type,
            joint_offset_map=new_joint_offset_map,
            tool_frame_map=new_tool_frame_map,
            link_chain_data=new_link_chain_data,
            link_chain_offsets=new_link_chain_offsets,
            joint_links_data=new_joint_links_data_tensor,
            joint_links_offsets=new_joint_links_offsets_tensor,
            joint_affects_endeffector=new_joint_affects_ee,
            tool_frames=desired_link_names,
            joint_limits=new_joint_limits,
            non_fixed_joint_names=new_joint_names,
            num_dof=len(needed_joints),
            joint_names=new_joint_names,
            link_spheres=torch.zeros((1, 4), device=device, dtype=dtype),  # No collision spheres
            link_sphere_idx_map=torch.zeros((1,), dtype=torch.int16, device=device),
            total_spheres=0,
            link_name_to_idx_map=new_link_name_to_idx_map,
            debug=config.debug,
            mesh_link_names=config.mesh_link_names,
            cspace=new_cspace,
            base_link=config.base_link,
            lock_jointstate=new_lock_jointstate,
            mimic_joints=new_mimic_joints,
            reference_link_spheres=None,
            link_masses_com=new_link_masses_com,
            link_inertias=new_link_inertias,
        )
