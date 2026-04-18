# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.logging import log_and_raise
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


@dataclass
class SeedIKSolverCfg:
    """Configuration for Seed IK Solver (Levenberg-Marquardt based)."""

    #: Robot configuration
    robot_config: RobotCfg

    #: Device and dtype configuration
    device_cfg: DeviceCfg = DeviceCfg()

    #: Maximum number of iterations
    max_iterations: int = 16

    inner_iterations: int = 4
    #: Position tolerance for convergence (meters)
    position_tolerance: float = 0.005

    #: Orientation tolerance for convergence (radians)
    orientation_tolerance: float = 0.05

    convergence_position_tolerance: float = 0.00001

    convergence_orientation_tolerance: float = 0.00001

    convergence_joint_limit_weight: float = 1.0

    #: Damping factor for Levenberg-Marquardt
    lambda_initial: float = 0.2

    #: Factor to increase damping when iteration fails
    lambda_factor: float = 2.0

    #: Maximum damping factor
    lambda_max: float = 1.0e10

    #: Minimum damping factor
    lambda_min: float = 1e-5  # 1e-8

    #: Joint limit margin (radians)
    joint_limit_margin: float = 0.001

    batch_success_threshold: float = 1.0

    #: Maximum step size in joint space
    max_step_size: float = 0.0

    #: Number of random seeds to try
    num_seeds: int = 1

    #: Weight for joint limit penalty
    joint_limit_weight: float = 1.0

    use_cuda_graph: bool = True

    use_backward: bool = True

    rho_min: float = 1e-3

    tile_threads: int = 32

    #: Seed for the sampler
    sampler_seed: int = 451

    max_problems_mini_batch: int = (
        200 * 512
    )  # allow to solve up to 1000 problems in a single mini batch

    start_cspace_dist_weight: float = 0.01

    position_weight: float = 1.0
    """Weight for position error in the pose cost."""

    orientation_weight: float = 1.0
    """Weight for orientation error in the pose cost."""

    velocity_weight: float = 0.0
    """Weight for velocity regularization residual. When > 0 and current_state
    provides dt, adds a residual penalising implied velocity
    ``v = (q - q_current) / dt``."""

    acceleration_weight: float = 0.0
    """Weight for acceleration regularization residual. When > 0 and current_state
    provides velocity and dt, adds a residual penalising implied acceleration
    ``a = ((q - q_current)/dt - v_current) / dt``. Disabled when velocity is
    not available on current_state."""

    @staticmethod
    def create(
        robot: Union[str, Dict, RobotCfg],
        device_cfg: DeviceCfg = DeviceCfg(),
        **kwargs,
    ) -> SeedIKSolverCfg:
        """Create SeedIKSolverCfg from robot configuration.

        Args:
            robot: Robot configuration file path, dictionary, or RobotCfg instance.
            device_cfg: Device and dtype configuration.
            **kwargs: Additional configuration parameters.

        Returns:
            Configured SeedIKSolverCfg instance.
        """
        if isinstance(robot, str):
            robot_cfg_dict = load_yaml(join_path(get_robot_configs_path(), robot))
            if "robot_cfg" not in robot_cfg_dict:
                robot_cfg_dict = {"robot_cfg": robot_cfg_dict}
            # robot_cfg_dict["robot_cfg"]["kinematics"]["collision_link_names"] = None
            # robot_cfg_dict["robot_cfg"]["kinematics"]["lock_joints"] = {}
            robot_config = RobotCfg.create(robot_cfg_dict, device_cfg)
        elif isinstance(robot, dict):
            robot_config = RobotCfg.create(robot, device_cfg)
        elif isinstance(robot, RobotCfg):
            # Use the provided RobotCfg instance directly for sharing
            robot_config = robot
        else:
            log_and_raise("robot must be string path, dict, or RobotCfg instance")

        config_dict = {"robot_config": robot_config, "device_cfg": device_cfg}

        config_dict.update(kwargs)

        return SeedIKSolverCfg(**config_dict)

    def __post_init__(self):
        if self.max_iterations < self.inner_iterations:
            log_and_raise(
                "max_iterations: {} must be greater than inner_iterations: {}".format(
                    self.max_iterations, self.inner_iterations
                )
            )
        if self.max_iterations % self.inner_iterations != 0:
            log_and_raise(
                "max_iterations: {} must be divisible by inner_iterations: {}".format(
                    self.max_iterations, self.inner_iterations
                )
            )

