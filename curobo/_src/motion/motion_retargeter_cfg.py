# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Configuration for :class:`~curobo._src.motion.motion_retargeter.MotionRetargeter`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class MotionRetargeterCfg:
    """Configuration for :class:`~curobo._src.motion.motion_retargeter.MotionRetargeter`.

    Use :meth:`create` to construct from simple arguments.
    """

    #: Robot config YAML filename or dict.
    robot: Union[str, Dict[str, Any]]

    #: Dict of link name to :class:`~curobo._src.cost.tool_pose_criteria.ToolPoseCriteria`.
    #: Each entry sets independent position (``xyz``) and rotation (``rpy``) weights for a
    #: tracked link. Use high weights (1.0) on feet/pelvis for balance and lower weights on
    #: mid-chain links like shoulders.
    tool_pose_criteria: Dict[str, ToolPoseCriteria]

    #: Batch size: number of clips retargeted in parallel on GPU. Shorter clips are padded
    #: by repeating the last frame's target. Increase to saturate GPU throughput when
    #: retargeting many clips.
    num_envs: int = 1

    #: Use MPC for frames 1+ instead of warm-started IK. MPC produces smoother
    #: trajectories (acceleration/jerk costs) but is 2-4x slower per frame.
    use_mpc: bool = False

    #: Enable self-collision avoidance. Disabling speeds up solving but may produce
    #: interpenetrating limbs.
    self_collision_check: bool = True

    #: Scene config (YAML or dict) for environment collision checking (ground plane,
    #: obstacles). ``None`` disables scene collision.
    scene_model: Optional[Union[str, Dict[str, Any]]] = None

    #: Time step (seconds) for velocity-limited IK or MPC. Decreasing enforces tighter
    #: velocity limits, producing smoother but potentially less accurate tracking.
    #: Increasing allows larger per-frame jumps.
    optimization_dt: float = 0.05

    #: Random seeds for frame-0 global IK. Increasing explores more of the configuration
    #: space (better initial pose) at the cost of longer first-frame solve time.
    #: Decreasing speeds up initialization but may land in a poor local minimum.
    num_seeds_global: int = 64

    #: Position convergence tolerance in meters. The solver stops early if all links are
    #: within this threshold. Tightening (e.g. 0.001) improves accuracy but may use more
    #: iterations. Loosening (e.g. 0.01) speeds up solving.
    position_tolerance: float = 0.005

    #: Orientation convergence tolerance in radians. Same trade-off as
    #: ``position_tolerance``.
    orientation_tolerance: float = 0.05

    #: Device and dtype configuration.
    device_cfg: DeviceCfg = field(default_factory=DeviceCfg)

    #: Load collision spheres for the robot. Auto-disabled when both
    #: ``self_collision_check=False`` and ``scene_model=None``. Set to ``False`` explicitly
    #: for faster initialization when collision checking is not needed.
    load_collision_spheres: bool = True

    #: Override the IK optimizer YAML config. The default runs 200 L-BFGS iterations.
    ik_optimizer_configs: List[Union[str, Dict[str, Any]]] = field(
        default_factory=lambda: ["ik/lbfgs_retarget_ik.yml"]
    )

    #: Override the MPC optimizer YAML config. The default runs 100 L-BFGS iterations.
    #: Ignored when ``use_mpc=False``.
    mpc_optimizer_configs: List[Union[str, Dict[str, Any]]] = field(
        default_factory=lambda: ["mpc/lbfgs_retarget_mpc.yml"]
    )

    #: Seeds for warm-started local IK. Ignored when ``use_mpc=True``.
    num_seeds_local: int = 1

    #: B-spline control points for MPC trajectory parameterization. More points allow
    #: the trajectory to represent higher-frequency motion but increase the optimization
    #: problem size. Ignored when ``use_mpc=False``.
    num_control_points: Optional[int] = None

    #: MPC optimization steps per input target frame. Increasing improves tracking quality
    #: and smoothness. Decreasing speeds up solving but pose reaching error will be higher.
    #: Ignored when ``use_mpc=False``.
    steps_per_target: int = 8

    #: Penalizes ``(q - q_prev) / dt``. ``None`` uses the YAML default (0.001 for IK).
    #: Increasing produces smoother joint velocities but the solver may lag behind
    #: fast-moving targets.
    velocity_regularization_weight: Optional[float] = None

    #: Penalizes ``(v - v_prev) / dt``. ``None`` uses the YAML default (0.01 for IK).
    #: Increasing produces smoother accelerations, particularly impactful in MPC mode.
    acceleration_regularization_weight: Optional[float] = None

    #: Activation distance [m] for collision cost. Collision penalties begin
    #: ramping when sphere surfaces are within this distance. Increasing creates a
    #: wider safety margin around obstacles but may reduce tracking accuracy in
    #: tight spaces.
    collision_activation_distance: float = 0.01

    #: L-BFGS iterations for frame-0 global IK (cold start, many seeds).
    #: ``None`` uses the optimizer YAML default (200 for ``lbfgs_retarget_ik.yml``).
    #: Increasing explores more thoroughly at the cost of longer first-frame time.
    global_ik_num_iters: Optional[int] = None

    #: L-BFGS iterations for warm-started local IK (frames 1+).
    #: ``None`` uses the optimizer YAML default (200 for ``lbfgs_retarget_ik.yml``).
    #: Can often be reduced (e.g. 100) since warm-starting provides a good seed.
    #: Ignored when ``use_mpc=True``.
    local_ik_num_iters: Optional[int] = None

    #: Optimizer iterations for warm-started MPC steps (frames 1+).
    #: Ignored when ``use_mpc=False``.
    mpc_warm_start_num_iters: int = 100

    #: Optimizer iterations for the first MPC step (cold start, no warm start).
    #: Should be >= ``mpc_warm_start_num_iters``. Ignored when ``use_mpc=False``.
    mpc_cold_start_num_iters: int = 300

    @property
    def tool_frames(self) -> List[str]:
        """Ordered list of tool frame link names (derived from criteria keys)."""
        return list(self.tool_pose_criteria.keys())

    @staticmethod
    def create(
        robot: Union[str, Dict[str, Any]],
        tool_pose_criteria: Dict[str, ToolPoseCriteria],
        num_envs: int = 1,
        use_mpc: bool = False,
        self_collision_check: bool = True,
        scene_model: Optional[Union[str, Dict[str, Any]]] = None,
        optimization_dt: float = 0.05,
        num_seeds_global: int = 64,
        load_collision_spheres: bool = True,
        ik_optimizer_configs: Optional[List[Union[str, Dict[str, Any]]]] = None,
        mpc_optimizer_configs: Optional[List[Union[str, Dict[str, Any]]]] = None,
        num_seeds_local: int = 1,
        num_control_points: Optional[int] = 12,
        steps_per_target: int = 4,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        device_cfg: DeviceCfg = DeviceCfg(),
        velocity_regularization_weight: Optional[float] = None,
        acceleration_regularization_weight: Optional[float] = None,
        collision_activation_distance: float = 0.01,
        global_ik_num_iters: Optional[int] = None,
        local_ik_num_iters: Optional[int] = None,
        mpc_warm_start_num_iters: int = 100,
        mpc_cold_start_num_iters: int = 300,
    ) -> MotionRetargeterCfg:
        """Create a MotionRetargeterCfg from simple arguments."""
        return MotionRetargeterCfg(
            robot=robot,
            tool_pose_criteria=tool_pose_criteria,
            num_envs=num_envs,
            use_mpc=use_mpc,
            self_collision_check=self_collision_check,
            scene_model=scene_model,
            optimization_dt=optimization_dt,
            num_seeds_global=num_seeds_global,
            load_collision_spheres=load_collision_spheres,
            ik_optimizer_configs=(
                ik_optimizer_configs
                if ik_optimizer_configs is not None
                else ["ik/lbfgs_retarget_ik.yml"]
            ),
            mpc_optimizer_configs=(
                mpc_optimizer_configs
                if mpc_optimizer_configs is not None
                else ["mpc/lbfgs_retarget_mpc.yml"]
            ),
            num_seeds_local=num_seeds_local,
            num_control_points=num_control_points,
            steps_per_target=steps_per_target,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            device_cfg=device_cfg,
            velocity_regularization_weight=velocity_regularization_weight,
            acceleration_regularization_weight=acceleration_regularization_weight,
            collision_activation_distance=collision_activation_distance,
            global_ik_num_iters=global_ik_num_iters,
            local_ik_num_iters=local_ik_num_iters,
            mpc_warm_start_num_iters=mpc_warm_start_num_iters,
            mpc_cold_start_num_iters=mpc_cold_start_num_iters,
        )

    def __post_init__(self):
        if not self.self_collision_check and self.scene_model is None:
            self.load_collision_spheres = False
        if self.self_collision_check:
            if not self.load_collision_spheres:
                log_and_raise("load_collision_spheres be True when self_collision_check==True")
        if self.scene_model is not None:
            if not self.load_collision_spheres:
                log_and_raise("load_collision_spheres must be True when scene_model is not None")
