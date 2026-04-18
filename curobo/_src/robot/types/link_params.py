# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Link parameter definitions for robot kinematics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from curobo._src.robot.types.joint_types import JointType
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise


@dataclass
class LinkParams:
    """Parameters of a link in the kinematic tree."""

    link_name: str
    joint_name: str
    joint_type: JointType
    fixed_transform: np.ndarray
    parent_link_name: Optional[str] = None
    child_link_name: Optional[str] = None
    joint_limits: Optional[List[float]] = None
    joint_axis: Optional[np.ndarray] = None
    joint_id: Optional[int] = None
    joint_velocity_limits: List[float] = field(default_factory=lambda: [-2.0, 2.0])
    joint_offset: List[float] = field(default_factory=lambda: [1.0, 0.0])
    mimic_joint_name: Optional[str] = None
    joint_effort_limit: List[float] = field(default_factory=lambda: [10000.0])
    link_mass: float = 0.01
    link_com: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    link_inertia: np.ndarray = field(
        default_factory=lambda: np.array([1e-4, 1e-4, 1e-4, 0.0, 0.0, 0.0])
    )

    @staticmethod
    def create(dict_data: Dict[str, Any]) -> LinkParams:
        """Create a LinkParams from a dictionary.

        Args:
            dict_data: Dictionary containing link parameters.

        Returns:
            LinkParams: Link parameters object.
        """
        dict_data["joint_type"] = JointType[dict_data["joint_type"]]
        dict_data["fixed_transform"] = (
            Pose.from_list(dict_data["fixed_transform"], device_cfg=DeviceCfg())
            .get_numpy_affine_matrix()
            .reshape(3, 4)
        )

        return LinkParams(**dict_data)

    def __post_init__(self):
        if self.fixed_transform.shape != (3, 4):
            log_and_raise(
                f"fixed_transform shape does not match: {self.fixed_transform.shape} != (3, 4)"
            )

    def get_link_com_and_mass(self) -> np.ndarray:
        """Get link center of mass and mass."""
        if self.link_com.shape != (3,):
            log_and_raise(f"link_com shape does not match: {self.link_com.shape} != (3,)")
        return np.concatenate([self.link_com, np.array([self.link_mass])])

