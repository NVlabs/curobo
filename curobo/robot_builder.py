# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Public API for building robot model configurations.

This module provides a simple interface for creating cuRobo robot configurations
from URDF files. It re-exports the main builder and debugger classes from the
internal implementation.

Example:
    Building a new robot model::

        >>> from curobo.robot_builder import RobotBuilder
        >>>
        >>> builder = RobotBuilder("robot.urdf", "assets/")
        >>> builder.fit_collision_spheres()
        >>> builder.compute_collision_matrix()
        >>> config = builder.build()
        >>> builder.save(config, "my_robot.yml")

    Editing an existing robot model::

        >>> builder = RobotBuilder.from_config("franka.yml")
        >>> builder.refit_link_spheres("panda_hand", sphere_density=3.0)
        >>> config = builder.build()
        >>> builder.save(config, "franka_updated.yml")

    Debugging collision issues::

        >>> from curobo.robot_builder import RobotDebugger
        >>>
        >>> # Load from YAML
        >>> debugger = RobotDebugger("robot.yml")
        >>>
        >>> # Or load from XRDF
        >>> debugger = RobotDebugger.from_xrdf("robot.xrdf", "robot.urdf", "assets/")
        >>>
        >>> result = debugger.check_retract_collision()
        >>> if result["has_collision"]:
        ...     print(f"Found {result['num_colliding_pairs']} colliding pairs")

    Exporting to XRDF format::

        >>> config = builder.build()
        >>> builder.save(config, "robot.yml")  # cuRobo YAML format
        >>> builder.save_xrdf(config, "robot.xrdf")  # XRDF format for Isaac Sim
"""

from curobo._src.robot.builder.builder_robot import RobotBuilder
from curobo._src.robot.builder.debugger_robot import RobotDebugger

__all__ = ["RobotBuilder", "RobotDebugger"]
