# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tools for debugging robot collision configurations.

This module provides :class:`RobotDebugger` for diagnosing self-collision issues
in robot configurations.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.state.state_joint import JointState
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.types.content_path import ContentPath
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_info, log_warn
from curobo._src.util.sampling.sample_buffer import SampleBuffer
from curobo._src.util.viser_visualizer import ViserVisualizer
from curobo._src.util.xrdf_util import convert_xrdf_to_curobo
from curobo._src.util_file import load_yaml


class RobotDebugger:
    """Debug robot model collision configurations.

    This class provides tools for diagnosing self-collision issues in robot
    configurations. Use it to:

    - Check for collisions at specific joint configurations
    - Identify problematic link pairs
    - Analyze collision statistics across random samples
    - Visualize collision states
    - Inspect collision matrix properties

    Example:
        Load from cuRobo YAML::

            >>> from curobo._src.robot.builder.debugger_robot import RobotDebugger
            >>>
            >>> # Load robot config
            >>> debugger = RobotDebugger("franka.yml")
            >>>
            >>> # Check default joint configuration
            >>> result = debugger.check_default_joint_configuration_collision()
            >>> if result["has_collision"]:
            ...     print(f"Found {result['num_colliding_pairs']} colliding pairs:")
            ...     for pair in result["colliding_pairs"]:
            ...         print(f"  {pair[0]} <-> {pair[1]}")

        Load from XRDF::

            >>> # Load from XRDF format
            >>> debugger = RobotDebugger.from_xrdf(
            ...     "robot.xrdf",
            ...     "robot.urdf",
            ...     "assets/"
            ... )
            >>> result = debugger.check_default_joint_configuration_collision()

        Check specific configuration::

            >>> q = [0, 0, 0, -1.5, 0, 1.8, 0]  # Joint angles
            >>> result = debugger.check_collision_at_config(q)
            >>>
            >>> # Print collision matrix statistics
            >>> debugger.print_collision_matrix_stats()
    """

    def __init__(
        self,
        config_path: str,
        device_cfg: Optional[DeviceCfg] = None,
    ):
        """Initialize debugger with cuRobo robot configuration (YAML format).

        Args:
            config_path: Path to robot configuration .yml file.
            device_cfg: Device configuration. Defaults to CUDA:0.

        Example:
            >>> debugger = RobotDebugger("robot.yml")
            >>> result = debugger.check_default_joint_configuration_collision()

        Note:
            For XRDF files, use :meth:`from_xrdf` instead.
        """
        self.config_path = config_path
        self.device_cfg = device_cfg or DeviceCfg()

        # Load configuration (cuRobo YAML format)
        config_data = load_yaml(config_path)
        if "robot_cfg" in config_data:
            config_data = config_data["robot_cfg"]

        # Create robot model
        self._robot_config = KinematicsCfg.from_data_dict(
            config_data["kinematics"], device_cfg=self.device_cfg
        )
        self._robot_model = Kinematics(self._robot_config)

        # Create collision cost checker
        self._collision_cost = SelfCollisionCost(
            SelfCollisionCostCfg(
                weight=1.0,
                self_collision_kin_config=self._robot_config.self_collision_config,
                store_pair_distance=True,
            )
        )

        log_info(f"Initialized debugger from cuRobo YAML: {config_path}")

    @classmethod
    def from_xrdf(
        cls,
        xrdf_path: str,
        urdf_path: Optional[str] = None,
        asset_path: str = "",
        device_cfg: Optional[DeviceCfg] = None,
    ) -> "RobotDebugger":
        """Initialize debugger from XRDF robot configuration.

        XRDF (eXtended Robot Description Format) is NVIDIA's format for robot
        descriptions. This method converts XRDF to cuRobo format and creates
        a debugger instance.

        Args:
            xrdf_path: Path to XRDF file.
            urdf_path: Path to URDF file (required by XRDF). If None, attempts
                to find it automatically from XRDF content.
            asset_path: Path to mesh assets directory.
            device_cfg: Device configuration. Defaults to CUDA:0.

        Returns:
            RobotDebugger instance initialized from XRDF.

        Example:
            >>> debugger = RobotDebugger.from_xrdf(
            ...     "robot.xrdf",
            ...     "robot.urdf",
            ...     "assets/"
            ... )
            >>> result = debugger.check_default_joint_configuration_collision()

        Note:
            XRDF support in the debugger provides full collision checking
            functionality. All debugging methods are available when loading
            from XRDF.
        """
        device_cfg = device_cfg or DeviceCfg()

        # Load XRDF file
        xrdf_dict = load_yaml(xrdf_path)

        # Validate XRDF format
        if "format" not in xrdf_dict or xrdf_dict["format"] != "xrdf":
            log_warn(f"File may not be valid XRDF format: {xrdf_path}")

        # Create content path
        content_path = ContentPath(
            robot_xrdf_file=xrdf_path,
            robot_urdf_file=urdf_path,
            robot_asset_root_path=asset_path,
        )

        # Convert XRDF to cuRobo format
        log_info("Converting XRDF to cuRobo configuration...")
        try:
            config_data = convert_xrdf_to_curobo(
                content_path=content_path,
                input_xrdf_dict=xrdf_dict,
            )
        except Exception as e:
            log_warn(f"Error converting XRDF: {e}")
            raise ValueError(f"Failed to convert XRDF file: {e}")

        if "robot_cfg" in config_data:
            config_data = config_data["robot_cfg"]

        # Create instance using private method
        instance = cls.__new__(cls)
        instance.config_path = xrdf_path
        instance.device_cfg = device_cfg

        # Create robot model
        instance._robot_config = KinematicsCfg.from_data_dict(
            config_data["kinematics"], device_cfg=device_cfg
        )
        instance._robot_model = Kinematics(instance._robot_config)

        # Create collision cost checker
        instance._collision_cost = SelfCollisionCost(
            SelfCollisionCostCfg(
                weight=1.0,
                self_collision_kin_config=instance._robot_config.self_collision_config,
                store_pair_distance=True,
            )
        )

        log_info(f"Initialized debugger from XRDF: {xrdf_path}")

        return instance

    def check_default_joint_configuration_collision(self) -> Dict:
        """Check for self-collisions at default joint configuration.

        The default joint configuration is typically a "home" or "neutral" pose where
        the robot should NOT have any self-collisions. If collisions are found,
        they likely indicate incorrectly fitted spheres or missing collision
        ignore pairs.

        Returns:
            Dictionary with collision information:
                - has_collision (bool): Whether any collisions were detected
                - num_colliding_pairs (int): Number of colliding link pairs
                - colliding_pairs (List[Tuple[str, str]]): List of colliding link name pairs
                - max_penetration (float): Maximum penetration depth
                - distances (Dict[Tuple[str, str], float]): Penetration depth per link pair

        Example:
            >>> result = debugger.check_default_joint_configuration_collision()
            >>> if result["has_collision"]:
            ...     print(f"WARNING: {result['num_colliding_pairs']} pairs colliding!")
            ...     print(f"Max penetration: {result['max_penetration']:.4f}m")
            ...     for (link1, link2), dist in result["distances"].items():
            ...         print(f"  {link1} <-> {link2}: {dist:.4f}m")
        """
        default_joint_position = self._robot_model.default_joint_state.position
        return self.check_collision_at_config(default_joint_position)

    def check_collision_at_config(
        self,
        joint_position: Union[List[float], np.ndarray, torch.Tensor],
    ) -> Dict:
        """Check self-collisions at a specific joint configuration.

        Args:
            joint_position: Joint angles. Can be list, numpy array, or torch tensor.
                Must match robot's DOF.

        Returns:
            Dictionary with detailed collision information (same format as
            :meth:`check_default_joint_configuration_collision`).

        Raises:
            ValueError: If joint_position length doesn't match robot DOF.

        Example:
            >>> # Check specific configuration
            >>> q = [0, -0.5, 0, -1.5, 0, 1.0, 0.785]
            >>> result = debugger.check_collision_at_config(q)
            >>> print(f"Collisions: {result['has_collision']}")
        """
        # Convert to tensor
        if not isinstance(joint_position, torch.Tensor):
            joint_position = torch.tensor(
                joint_position, **self.device_cfg.as_torch_dict()
            )

        if joint_position.numel() != self._robot_config.dof:
            raise ValueError(
                f"joint_position must have {self._robot_config.dof} elements, "
                f"got {joint_position.numel()}"
            )

        joint_position = joint_position.view(1, -1)

        # Compute kinematics
        kin_state = self._robot_model.compute_kinematics(
            JointState.from_position(joint_position, joint_names=self._robot_model.joint_names)
        )
        robot_spheres = kin_state.robot_spheres

        # Check collisions
        self._collision_cost.setup_batch_tensors(1, 1)
        collision_cost = self._collision_cost.forward(robot_spheres)
        pair_distances = self._collision_cost._pair_distance.view(-1)

        # Positive distance = collision
        colliding_mask = pair_distances > 0
        has_collision = torch.any(colliding_mask).item()

        result = {
            "has_collision": has_collision,
            "num_colliding_pairs": torch.sum(colliding_mask).item(),
            "colliding_pairs": [],
            "max_penetration": 0.0,
            "distances": {},
        }

        if has_collision:
            # Map sphere pairs to link pairs
            collision_pairs = self._robot_config.self_collision_config.collision_pairs
            colliding_indices = collision_pairs[colliding_mask]

            # Get link names for colliding pairs
            link_sphere_idx_map = (
                self._robot_config.kinematics_config.link_sphere_idx_map
            )
            colliding_link_pairs = link_sphere_idx_map[
                colliding_indices.to(dtype=torch.int32)
            ]

            # Unique link pairs
            unique_link_pairs = torch.unique(colliding_link_pairs, dim=0)

            # Get link index to name mapping
            idx_to_name = {
                v: k
                for k, v in self._robot_config.kinematics_config.link_name_to_idx_map.items()
            }

            # Build result
            unique_link_pairs_cpu = unique_link_pairs.cpu().numpy()

            for idx1, idx2 in unique_link_pairs_cpu:
                link_pair = (idx_to_name[int(idx1)], idx_to_name[int(idx2)])
                result["colliding_pairs"].append(link_pair)

                # Find max penetration for this link pair
                mask = (colliding_link_pairs[:, 0] == idx1) & (
                    colliding_link_pairs[:, 1] == idx2
                )
                pair_penetrations = pair_distances[colliding_mask][mask]
                if len(pair_penetrations) > 0:
                    result["distances"][link_pair] = float(
                        torch.max(pair_penetrations).item()
                    )

            result["max_penetration"] = float(torch.max(pair_distances[colliding_mask]).item())

        return result

    def sample_collision_checks(
        self,
        num_samples: int = 1000,
        batch_size: int = 100,
        seed: int = 42,
    ) -> Dict:
        """Sample random configurations and check for collisions.

        This helps identify which link pairs collide most frequently, which can
        indicate problematic sphere fitting or missing collision ignore pairs.

        Args:
            num_samples: Total number of configurations to sample.
            batch_size: Batch size for parallel collision checking.
            seed: Random seed for reproducibility.

        Returns:
            Dictionary with collision statistics:
                - total_samples (int): Number of configurations sampled
                - collision_count (int): Number of configs with collisions
                - collision_rate (float): Percentage of configs with collisions
                - frequent_collisions (List[Tuple[Tuple[str, str], int]]):
                    List of (link_pair, count) sorted by frequency

        Example:
            >>> stats = debugger.sample_collision_checks(num_samples=5000)
            >>> print(f"Collision rate: {stats['collision_rate']:.1f}%")
            >>> print("Most frequent collisions:")
            >>> for (link1, link2), count in stats["frequent_collisions"][:10]:
            ...     print(f"  {link1} <-> {link2}: {count} times")
        """
        # Get joint limits
        joint_limits = self._robot_model.config.get_joint_limits()

        # Create sample generator
        sample_generator = SampleBuffer.create_halton_sample_buffer(
            self._robot_config.dof,
            device_cfg=self.device_cfg,
            low_bounds=joint_limits.position_lower_limits,
            up_bounds=joint_limits.position_upper_limits,
            seed=seed,
            store_buffer=None,
        )

        # Track collisions
        total_collisions = 0
        link_pair_collisions: Dict[Tuple[str, str], int] = {}

        # Get link index to name mapping
        idx_to_name = {
            v: k
            for k, v in self._robot_config.kinematics_config.link_name_to_idx_map.items()
        }

        log_info(f"Sampling {num_samples} configurations...")

        num_batches = (num_samples + batch_size - 1) // batch_size
        for i in range(num_batches):
            actual_batch_size = min(batch_size, num_samples - i * batch_size)
            samples = sample_generator.get_samples(actual_batch_size, bounded=True)

            # Compute collisions
            kin_state = self._robot_model.compute_kinematics(
                JointState.from_position(samples.contiguous(), joint_names=self._robot_model.joint_names)
            )
            robot_spheres = kin_state.robot_spheres
            self._collision_cost.setup_batch_tensors(actual_batch_size, 1)
            out_distance = self._collision_cost.forward(robot_spheres)
            pair_distances = self._collision_cost._pair_distance.view(
                actual_batch_size, -1
            )

            # Count collisions
            colliding_mask = pair_distances > 0  # [batch, num_pairs]
            configs_with_collisions = torch.any(colliding_mask, dim=1)
            total_collisions += torch.sum(configs_with_collisions).item()

            # Track link pair frequencies
            if torch.any(colliding_mask):
                collision_pairs = (
                    self._robot_config.self_collision_config.collision_pairs
                )
                link_sphere_idx_map = (
                    self._robot_config.kinematics_config.link_sphere_idx_map
                )

                for batch_idx in range(actual_batch_size):
                    if configs_with_collisions[batch_idx]:
                        colliding_pairs = collision_pairs[
                            colliding_mask[batch_idx], :
                        ]
                        colliding_link_pairs = link_sphere_idx_map[
                            colliding_pairs.to(dtype=torch.int32)
                        ]
                        unique_link_pairs = torch.unique(colliding_link_pairs, dim=0)

                        for idx1, idx2 in unique_link_pairs.cpu().numpy():
                            link_pair = (
                                idx_to_name[int(idx1)],
                                idx_to_name[int(idx2)],
                            )
                            link_pair_collisions[link_pair] = (
                                link_pair_collisions.get(link_pair, 0) + 1
                            )

        # Sort by frequency
        frequent_collisions = sorted(
            link_pair_collisions.items(), key=lambda x: x[1], reverse=True
        )

        return {
            "total_samples": num_samples,
            "collision_count": total_collisions,
            "collision_rate": (total_collisions / num_samples) * 100,
            "frequent_collisions": frequent_collisions,
        }

    def find_never_colliding_pairs(
        self,
        num_samples: int = 10000,
        batch_size: int = 10000,
        seed: int = 345,
    ) -> List[Tuple[str, str]]:
        """Find link pairs that never collide in sampled configurations.

        These pairs can potentially be added to the collision ignore list to
        improve collision checking performance.

        Args:
            num_samples: Number of configurations to sample.
            batch_size: Batch size for collision checking.
            seed: Random seed.

        Returns:
            List of link name pairs that never collided across all samples.

        Example:
            >>> never_colliding = debugger.find_never_colliding_pairs(num_samples=20000)
            >>> print(f"Found {len(never_colliding)} pairs that never collide:")
            >>> for link1, link2 in never_colliding[:10]:
            ...     print(f"  {link1} <-> {link2}")
            >>> # These can be added to collision_ignore in the config
        """
        # Get joint limits
        joint_limits = self._robot_model.config.get_joint_limits()

        # Create sample generator
        sample_generator = SampleBuffer.create_halton_sample_buffer(
            self._robot_config.dof,
            device_cfg=self.device_cfg,
            low_bounds=joint_limits.position_lower_limits - 0.1,
            up_bounds=joint_limits.position_upper_limits + 0.1,
            seed=seed,
            store_buffer=None,
        )

        # Group collision pairs by link pairs
        collision_pairs = self._robot_config.self_collision_config.collision_pairs
        link_sphere_idx_map = (
            self._robot_config.kinematics_config.link_sphere_idx_map
        )
        collision_link_pairs = link_sphere_idx_map[collision_pairs.to(dtype=torch.int32)]

        # Get unique link pairs
        unique_link_pairs, inverse_indices = torch.unique(
            collision_link_pairs, dim=0, return_inverse=True
        )

        # Track which pairs never collide
        link_pairs_never_collide = torch.ones(
            len(unique_link_pairs), dtype=torch.bool, device=collision_pairs.device
        )

        log_info(f"Sampling {num_samples} configurations to find never-colliding pairs...")

        # Sample and check
        samples = sample_generator.get_samples(batch_size, bounded=True)
        kin_state = self._robot_model.compute_kinematics(
            JointState.from_position(samples.contiguous(), joint_names=self._robot_model.joint_names)
        )
        robot_spheres = kin_state.robot_spheres
        self._collision_cost.setup_batch_tensors(batch_size, 1)
        out_distance = self._collision_cost.forward(robot_spheres)
        pair_distances = self._collision_cost._pair_distance.view(batch_size, -1)

        # Check each unique link pair
        for link_pair_idx in range(len(unique_link_pairs)):
            if not link_pairs_never_collide[link_pair_idx]:
                continue

            # Get sphere pairs for this link pair
            sphere_pair_mask = inverse_indices == link_pair_idx
            link_pair_distances = pair_distances[:, sphere_pair_mask]

            # If any collision detected, mark as colliding
            if torch.any(link_pair_distances > 0):
                link_pairs_never_collide[link_pair_idx] = False

        # Convert to link names
        never_colliding_pairs = unique_link_pairs[link_pairs_never_collide]
        idx_to_name = {
            v: k
            for k, v in self._robot_config.kinematics_config.link_name_to_idx_map.items()
        }

        result = []
        for idx1, idx2 in never_colliding_pairs.cpu().numpy():
            result.append((idx_to_name[int(idx1)], idx_to_name[int(idx2)]))

        log_info(f"Found {len(result)} link pairs that never collide")

        return result

    def visualize_collision_at_config(
        self,
        joint_position: Union[List[float], np.ndarray, torch.Tensor],
        port: int = 8080,
    ) -> ViserVisualizer:
        """Visualize robot at specific configuration.

        Args:
            joint_position: Joint angles to visualize.
            port: Viser server port.

        Returns:
            ViserVisualizer instance.

        Example:
            >>> q = [0, -0.5, 0, -1.5, 0, 1.0, 0.785]
            >>> viser = debugger.visualize_collision_at_config(q)
            >>> # Opens browser visualization at http://localhost:8080
        """
        # Create robot content
        config_data = load_yaml(self.config_path)
        if "robot_cfg" in config_data:
            config_data = config_data["robot_cfg"]

        robot_content = ContentPath(robot_config_file=config_data)

        # Create visualization
        viser_visualizer = ViserVisualizer(
            content_path=robot_content,
            connect_ip="0.0.0.0",
            connect_port=port,
            add_control_frames=True,
            visualize_robot_spheres=True,
        )

        # Set joint configuration
        if not isinstance(joint_position, torch.Tensor):
            joint_position = torch.tensor(
                joint_position, **self.device_cfg.as_torch_dict()
            )

        log_info(f"Started visualization at http://localhost:{port}")

        return viser_visualizer

    def print_collision_matrix_stats(self) -> None:
        """Print statistics about the collision matrix.

        Shows information about total spheres, collision pairs being checked,
        and percentage of sphere pairs that are ignored.

        Example:
            >>> debugger.print_collision_matrix_stats()
            Collision Matrix Statistics:
              Total spheres: 245
              Total possible pairs: 29,890
              Checked pairs: 8,432
              Ignored pairs: 21,458
              Checking: 28.2% of all possible pairs
        """
        collision_config = self._robot_config.self_collision_config
        num_spheres = collision_config.num_spheres
        num_collision_pairs = collision_config.collision_pairs.shape[0]
        total_possible = (num_spheres * (num_spheres - 1)) // 2

        print("Collision Matrix Statistics:")
        print(f"  Total spheres: {num_spheres}")
        print(f"  Total possible pairs: {total_possible:,}")
        print(f"  Checked pairs: {num_collision_pairs:,}")
        print(f"  Ignored pairs: {total_possible - num_collision_pairs:,}")
        print(
            f"  Checking: {num_collision_pairs / total_possible * 100:.1f}% of all possible pairs"
        )

        # Additional info
        kinematics_config = self._robot_config.kinematics_config
        if hasattr(kinematics_config, "link_name_to_idx_map"):
            num_collision_links = len(
                set(
                    kinematics_config.link_sphere_idx_map[
                        collision_config.collision_pairs.flatten().to(dtype=torch.int32)
                    ]
                    .cpu()
                    .numpy()
                )
            )
            print(f"  Collision links: {num_collision_links}")

    @property
    def robot_config(self) -> KinematicsCfg:
        """Get the robot configuration being debugged."""
        return self._robot_config

    @property
    def robot_model(self) -> Kinematics:
        """Get the robot model instance."""
        return self._robot_model

