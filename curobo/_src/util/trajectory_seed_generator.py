# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Standard Library
from typing import Optional

# Third Party
import torch

# CuRobo
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_torch_jit_decorator


def interpolate_kernel(h, int_steps, device_cfg: DeviceCfg):
    mat = torch.zeros(
        ((h - 1) * (int_steps), h), device=device_cfg.device, dtype=device_cfg.dtype
    )
    # delta = torch.arange(0, int_steps, device=device_cfg.device, dtype=device_cfg.dtype) / (
    #    int_steps - 1
    # )
    delta = torch.linspace(0, 1, int_steps, device=device_cfg.device, dtype=device_cfg.dtype)
    rev_delta = delta.flip(0)
    for i in range(h - 1):
        mat[i * int_steps : i * int_steps + int_steps, i] = rev_delta
        mat[i * int_steps : i * int_steps + int_steps, i + 1] = delta
    return mat

class TrajectorySeedGenerator:
    def __init__(self, action_horizon: int, action_dim: int, device_cfg: DeviceCfg):
        self.device_cfg = device_cfg
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        interpolation_weights = interpolate_kernel(2, self.action_horizon, self.device_cfg)
        # Shape: (1, action_horizon, 2)
        self._interpolation_weights = interpolation_weights.unsqueeze(0).reshape(
            1, self.action_horizon, 2, 1
        )

    @get_torch_jit_decorator(dynamic=True, only_valid_for_compile=True)
    def generate_constant_seeds(self, constant_position: torch.Tensor, num_seeds: int):
        """Generate constant seeds for TrajOpt optimization where all points in the trajectory
        are identical to the constant_position.

        Args:
            constant_position: Position to use for all points in the trajectory.
                              Shape: [batch_size, action_dim]
            num_seeds: Number of seed trajectories to generate.

        Returns:
            Constant trajectory seeds with shape [batch_size, num_seeds, action_horizon, action_dim]
        """
        self._validate_constant_inputs(constant_position, num_seeds)

        constant_trajectory = (
            constant_position.unsqueeze(1).unsqueeze(2).repeat(1, num_seeds, self.action_horizon, 1)
        )
        return constant_trajectory

    @get_torch_jit_decorator(dynamic=True, only_valid_for_compile=True)
    def generate_interpolated_seeds(
        self, start_position: torch.Tensor, goal_position: torch.Tensor, num_seeds: int
    ):
        """Generate seeds for Batched TrajOpt optimization. This method generates seed trajectories
        with shape: [batch_size, num_seeds, action_horizon]

        Args:
            start_position: Starting position of the trajectory. Shape: [batch_size, action_dim]
            goal_position: Goal position of the trajectory. Shape: [batch_size, num_seeds, action_dim]
            num_seeds: Number of seed trajectories to generate.
        """
        self._validate_interpolation_inputs(start_position, goal_position, num_seeds)

        # Prepare start and end points for interpolation
        # Start state needs repeating to match number of interpolations
        start_position_repeated = start_position.unsqueeze(1).repeat(1, num_seeds, 1)

        interpolated_traj = self._interpolate_trajectory(start_position_repeated, goal_position)

        return interpolated_traj

    @get_torch_jit_decorator(dynamic=True, only_valid_for_compile=True)
    def _interpolate_trajectory(
        self, start_position_seeds: torch.Tensor, goal_position_seeds: torch.Tensor
    ):
        """Interpolate a trajectory between start and goal positions using the precomputed weights.

        Args:
            start_position_seeds: Starting positions, shape [batch_size, num_seeds, action_dim]
            goal_position_seeds: Goal positions, shape [batch_size, num_seeds, action_dim]

        Returns:
            Interpolated trajectory with shape [batch_size, num_seeds, action_horizon, action_dim]
        """
        batch_size, num_seeds, action_dim = start_position_seeds.shape

        start_flat = start_position_seeds.reshape(
            batch_size * num_seeds, 1, action_dim
        )  # [batch_size*num_seeds, action_dim]
        goal_flat = goal_position_seeds.reshape(
            batch_size * num_seeds, 1, action_dim
        )  # [batch_size*num_seeds, action_dim]

        # (batch*num_seeds, 2, action_dim)

        # Interpolate using precomputed interpolation weights
        # [ 1, action_horizon, 2] @ [batch_size * num_seeds, 1, action_dim]
        # Result shape: (batch_size * num_seeds, action_horizon, action_dim)
        interpolated_traj = (
            self._interpolation_weights[:, :, 0, :] * start_flat
            + self._interpolation_weights[:, :, 1, :] * goal_flat
        )

        interpolated_traj = interpolated_traj.view(
            batch_size, num_seeds, self.action_horizon, action_dim
        )

        return interpolated_traj

    @get_torch_jit_decorator(dynamic=True, only_valid_for_compile=True)
    def generate_deceleration_seeds(
        self,
        current_state: JointState,
        num_seeds: int,
        deceleration_time: Optional[float] = None,
        deceleration_profile: str = "exponential",
    ) -> torch.Tensor:
        """Generate deceleration seeds using acceleration space approach.

        Physics-based approach:
        1. Generate smooth acceleration profile that opposes current velocity
        2. Integrate acceleration to get velocity profile (ensures monotonic deceleration)
        3. Integrate velocity to get position profile (B-spline knots)
        4. Prevents oscillation through proper dynamics

        Args:
            current_state: Current robot state with position, velocity, acceleration
            num_seeds: Number of seed trajectories to generate
            deceleration_time: Time over which to decelerate (uses full horizon if None)
            deceleration_profile: Type of deceleration ("linear", "exponential", "smooth")

        Returns:
            Deceleration trajectory seeds [batch_size, num_seeds, action_horizon, action_dim]
        """
        self._validate_deceleration_inputs(current_state, num_seeds)

        batch_size = current_state.position.shape[0]

        # Get current motion state - use actual state, don't zero!
        current_pos = current_state.position  # [batch_size, action_dim]
        current_vel = (
            current_state.velocity
            if current_state.velocity is not None
            else torch.zeros_like(current_pos)
        )
        current_acc = (
            current_state.acceleration
            if current_state.acceleration is not None
            else torch.zeros_like(current_pos)
        )

        dt = current_state.dt[0] if current_state.dt.ndim > 0 else current_state.dt

        # Generate acceleration profile that opposes current velocity
        acceleration_profile = self._generate_deceleration_acceleration_profile(
            current_vel, current_acc, deceleration_profile, dt
        )

        # Double integration: acceleration → velocity → position
        position_trajectory = self._integrate_acceleration_to_trajectory(
            current_pos, current_vel, acceleration_profile, dt
        )

        # Repeat for multiple seeds
        trajectory = position_trajectory.unsqueeze(1).repeat(1, num_seeds, 1, 1)

        return trajectory

    def _generate_linear_deceleration_profile(self, decel_steps: int) -> torch.Tensor:
        """Generate linear deceleration profile from 1.0 to 0.0"""
        if decel_steps <= 0:
            return torch.tensor([], device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        profile = torch.linspace(
            1.0, 0.0, decel_steps, device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )
        return profile

    def _generate_exponential_deceleration_profile(self, decel_steps: int) -> torch.Tensor:
        """Generate exponential deceleration profile"""
        if decel_steps <= 0:
            return torch.tensor([], device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        t = torch.linspace(
            0, 1, decel_steps, device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )
        profile = torch.exp(-3.0 * t)  # Exponential decay with rate 3.0
        return profile

    def _generate_smooth_deceleration_profile(self, decel_steps: int) -> torch.Tensor:
        """Generate smooth (cosine-based) deceleration profile"""
        if decel_steps <= 0:
            return torch.tensor([], device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        t = torch.linspace(
            0, torch.pi, decel_steps, device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )
        profile = (torch.cos(t) + 1.0) / 2.0  # Smooth cosine deceleration
        return profile

    def _generate_deceleration_acceleration_profile(
        self,
        current_vel: torch.Tensor,
        current_acc: torch.Tensor,
        deceleration_profile: str,
        dt: float,
    ) -> torch.Tensor:
        """Generate acceleration profile that smoothly transitions from current acceleration
        to deceleration that opposes current velocity direction.

        Physics-aware approach:
        1. Analyze current acceleration relative to velocity
        2. Smoothly transition from current acceleration to desired deceleration
        3. Avoid abrupt acceleration changes
        4. Take advantage of existing beneficial deceleration

        Args:
            current_vel: Current velocity [batch_size, action_dim]
            current_acc: Current acceleration [batch_size, action_dim]
            deceleration_profile: Type of deceleration curve
            dt: Time step

        Returns:
            Acceleration profile [batch_size, action_horizon, action_dim]
        """
        batch_size, action_dim = current_vel.shape

        # Maximum deceleration magnitude (adjust based on robot capabilities)
        max_deceleration = 10.0  # rad/s² or m/s²

        # Determine desired deceleration direction (always oppose current velocity)
        velocity_direction = torch.sign(current_vel)  # [batch_size, action_dim]
        desired_decel_direction = -velocity_direction  # Always oppose velocity

        # Handle zero velocity case (no deceleration needed)
        velocity_magnitude = torch.abs(current_vel)
        needs_deceleration = velocity_magnitude > 1e-6

        # Analyze current acceleration relative to velocity
        # Is current acceleration already helping with deceleration?
        current_acc_direction = torch.sign(current_acc)
        acc_opposes_vel = (current_acc_direction == desired_decel_direction) & needs_deceleration

        # Calculate how much current acceleration contributes to deceleration
        current_decel_magnitude = torch.where(
            acc_opposes_vel,
            torch.abs(current_acc),  # Current acceleration is helpful
            torch.zeros_like(current_acc),  # Current acceleration doesn't help
        )

        # Generate target magnitude profile over time
        time_steps = torch.linspace(
            0, 1, self.action_horizon, device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )

        if deceleration_profile == "linear":
            # Linear decay: max → 0
            target_magnitude_profile = max_deceleration * (1.0 - time_steps)
        elif deceleration_profile == "exponential":
            # Exponential decay: fast initial deceleration, then gradual
            decay_rate = 3.0
            target_magnitude_profile = max_deceleration * torch.exp(-decay_rate * time_steps)
        elif deceleration_profile == "smooth":
            # Smooth cosine decay
            target_magnitude_profile = (
                max_deceleration * (torch.cos(time_steps * torch.pi) + 1.0) / 2.0
            )
        else:
            # Default to exponential
            target_magnitude_profile = max_deceleration * torch.exp(-3.0 * time_steps)

        # Create smooth transition from current acceleration to target profile
        transition_steps = min(5, self.action_horizon // 3)  # Transition over first few steps

        acceleration_profile = []
        for t in range(self.action_horizon):
            if t < transition_steps:
                # Smooth transition from current acceleration to target
                blend_factor = t / max(transition_steps - 1, 1)  # 0 to 1

                # Target acceleration for this timestep
                target_accel = desired_decel_direction * target_magnitude_profile[t]

                # Blend from current acceleration to target acceleration
                blended_accel = (1.0 - blend_factor) * current_acc + blend_factor * target_accel

                # Ensure we don't exceed maximum deceleration
                accel_magnitude = torch.abs(blended_accel)
                accel_direction = torch.sign(blended_accel)
                clamped_magnitude = torch.clamp(accel_magnitude, 0, max_deceleration)
                final_accel = accel_direction * clamped_magnitude

            else:
                # Pure target acceleration profile after transition
                final_accel = desired_decel_direction * target_magnitude_profile[t]

            # Zero out acceleration for joints with no velocity
            final_accel = torch.where(
                needs_deceleration, final_accel, torch.zeros_like(final_accel)
            )

            acceleration_profile.append(final_accel)

        # Stack to create [batch_size, action_horizon, action_dim]
        acceleration_profile = torch.stack(acceleration_profile, dim=1)

        return acceleration_profile

    def _integrate_acceleration_to_trajectory(
        self,
        current_pos: torch.Tensor,
        current_vel: torch.Tensor,
        acceleration_profile: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Double integration: acceleration → velocity → position

        Ensures physically consistent trajectory with guaranteed monotonic deceleration.

        Args:
            current_pos: Initial position [batch_size, action_dim]
            current_vel: Initial velocity [batch_size, action_dim]
            acceleration_profile: Acceleration over time [batch_size, action_horizon, action_dim]
            dt: Time step

        Returns:
            Position trajectory [batch_size, action_horizon, action_dim]
        """
        batch_size, action_dim = current_pos.shape

        positions = [current_pos]
        velocities = [current_vel]

        for t in range(1, self.action_horizon):
            # Step 1: Integrate acceleration to get velocity
            # v(t+1) = v(t) + a(t) * dt
            prev_vel = velocities[-1]
            accel_t = acceleration_profile[:, t - 1, :]  # Use previous timestep acceleration

            new_vel = prev_vel + accel_t * dt

            # Step 2: Safety check - prevent velocity direction reversal
            # If velocity changes sign, clamp to zero (robot has stopped)
            initial_vel_sign = torch.sign(current_vel)
            current_vel_sign = torch.sign(new_vel)

            # Zero velocity if direction changed or if very small
            direction_changed = (initial_vel_sign != current_vel_sign) & (
                torch.abs(current_vel) > 1e-6
            )
            velocity_too_small = torch.abs(new_vel) < 1e-6

            new_vel = torch.where(
                direction_changed | velocity_too_small, torch.zeros_like(new_vel), new_vel
            )

            # Step 3: Integrate velocity to get position
            # p(t+1) = p(t) + v(t) * dt  (use previous velocity for integration)
            prev_pos = positions[-1]
            new_pos = prev_pos + prev_vel * dt

            velocities.append(new_vel)
            positions.append(new_pos)

        # Stack to create trajectory: [batch_size, action_horizon, action_dim]
        position_trajectory = torch.stack(positions, dim=1)

        return position_trajectory

    def _validate_interpolation_inputs(
        self, start_position: torch.Tensor, goal_position: torch.Tensor, num_seeds: int
    ):
        """Validate the inputs for interpolated seed generation.

        Args:
            start_position: Starting position of the trajectory. Shape: [batch_size, action_dim]
            goal_position: Goal position of the trajectory. Shape: [batch_size, num_seeds, action_dim]
            num_seeds: Number of seed trajectories to generate.
        """
        if num_seeds < 1:
            log_and_raise("num_seeds must be greater than 0")
        if start_position.ndim != 2:
            log_and_raise(f"start_position must be a 2D tensor, got shape: {start_position.shape}")
        if goal_position.ndim != 3:
            log_and_raise(f"goal_position must be a 3D tensor, got shape: {goal_position.shape}")
        if start_position.shape[0] != goal_position.shape[0]:
            log_and_raise(
                "start_position must have a batch size of 1 or the same batch size as goal_position"
            )

        if start_position.shape[-1] != self.action_dim:
            log_and_raise("start_position and goal_position must have the same number of dimensions")
        if goal_position.shape[-1] != self.action_dim:
            log_and_raise("goal_position must have the same number of dimensions as start_position")

        if goal_position.shape[1] != num_seeds:
            log_and_raise("goal_position must have the same number of seeds as num_seeds")

    def _validate_constant_inputs(self, constant_position: torch.Tensor, num_seeds: int):
        """Validate the inputs for constant seed generation.

        Args:
            constant_position: Position to use for all points in the trajectory.
                              Shape: [batch_size, action_dim]
            num_seeds: Number of seed trajectories to generate.
        """
        if num_seeds < 1:
            log_and_raise("num_seeds must be greater than 0")
        if constant_position.ndim != 2:
            log_and_raise("constant_position must be a 2D tensor")
        if constant_position.shape[-1] != self.action_dim:
            log_and_raise("constant_position must have the same number of dimensions as action_dim")

    def _validate_deceleration_inputs(self, current_state: JointState, num_seeds: int):
        """Validate inputs for deceleration seed generation"""
        if num_seeds < 1:
            log_and_raise("num_seeds must be greater than 0")
        if current_state.position.ndim != 2:
            log_and_raise("current_state.position must be a 2D tensor")
        if current_state.position.shape[-1] != self.action_dim:
            log_and_raise("current_state.position must have the same number of dimensions as action_dim")
