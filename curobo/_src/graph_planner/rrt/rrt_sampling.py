# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape informed sampling for the bounded bidirectional RRT planner."""

from __future__ import annotations

# Standard Library
import math

# Third Party
import torch
from torch import nn
from torch.nn import functional as F

from curobo._src.types.device_cfg import DeviceCfg

# CuRobo
from curobo._src.util.logging import log_and_raise
from curobo._src.util.sampling.sample_buffer import SampleBuffer

__all__ = [
    "InformedRRTProposalGenerator",
    "UniformRRTSampler",
]


class UniformRRTSampler:
    """Own the uniform Halton stream used by RRT planning and warmup."""

    def __init__(
        self,
        action_lows: torch.Tensor,
        action_highs: torch.Tensor,
        sample_rejection_ratio: int,
        seed: int,
        device_cfg: DeviceCfg,
        check_feasibility_fn,
    ) -> None:
        """Create a bounded Halton stream over the robot action limits."""
        self.action_dim = action_lows.numel()
        self.sample_rejection_ratio = sample_rejection_ratio
        self.check_feasibility_fn = check_feasibility_fn
        self.action_sample_generator = SampleBuffer.create_halton_sample_buffer(
            ndims=self.action_dim,
            device_cfg=device_cfg,
            up_bounds=action_highs,
            low_bounds=action_lows,
            seed=seed,
            cache_bounded_samples=True,
        )

    def generate_action_samples(self, n_samples: int, bounded: bool = True) -> torch.Tensor:
        """Generate one bounded or raw uniform sample batch."""
        return self.action_sample_generator.get_samples(n_samples, bounded=bounded)

    def generate_feasible_action_samples(self, num_samples: int) -> torch.Tensor:
        """Generate and filter the warmup samples needed by the planner."""
        candidate_count = num_samples + int(num_samples * self.sample_rejection_ratio)
        candidates = self.generate_action_samples(candidate_count, bounded=True)
        feasible = candidates[self.check_feasibility_fn(candidates).reshape(-1)]
        return feasible[:num_samples]

    def reset_seed(self) -> None:
        """Reset only the RRT Halton stream."""
        self.action_sample_generator.reset()


class UnitBallRRTTransform(nn.Module):
    """Map independent unit-cube coordinates into a filled unit ball."""

    def __init__(self, action_dim: int) -> None:
        """Store the fixed robot configuration-space dimension."""
        super().__init__()
        if action_dim < 1:
            log_and_raise("action_dim must be positive")
        self.action_dim = action_dim

    def forward(self, unit_cube: torch.Tensor) -> torch.Tensor:
        """Return volume-uniform ball points from ``[..., action_dim + 1]`` inputs."""
        gaussian = torch.erfinv(1.98 * unit_cube[..., : self.action_dim] - 0.99)
        gaussian = gaussian * 1.4142135623730951
        gaussian_norm = torch.linalg.vector_norm(gaussian, dim=-1, keepdim=True)
        first_axis = F.one_hot(
            torch.zeros((), device=unit_cube.device, dtype=torch.int64),
            num_classes=self.action_dim,
        ).to(dtype=unit_cube.dtype)
        direction = torch.where(
            gaussian_norm > 1.0e-12,
            gaussian / gaussian_norm.clamp_min(1.0e-12),
            first_axis,
        )
        radius = torch.clamp(
            unit_cube[..., self.action_dim : self.action_dim + 1],
            min=0.0,
            max=1.0,
        ).pow(1.0 / self.action_dim)
        return direction * radius


class InformedEllipsoidRRTTransform(nn.Module):
    """Replace fixed proposal slots with weighted prolate-hyperspheroid samples.

    The remaining slots are preserved byte-for-byte as uniform fallback samples.
    An informed point outside the robot joint limits also falls back to its
    corresponding uniform point instead of being clamped onto a limit.
    """

    def __init__(
        self,
        distance_weight: torch.Tensor,
        action_lows: torch.Tensor,
        action_highs: torch.Tensor,
        informed_slots: int,
        informed_start: int = 0,
    ) -> None:
        """Store robot-specific weighted-space geometry and joint limits."""
        super().__init__()
        weight = distance_weight.detach().clone().reshape(-1)
        lows = action_lows.detach().clone().reshape(-1)
        highs = action_highs.detach().clone().reshape(-1)
        if weight.numel() < 1 or lows.shape != weight.shape or highs.shape != weight.shape:
            log_and_raise("weights and action bounds must be matching non-empty vectors")
        if bool(torch.any(~torch.isfinite(weight)).item()) or bool(
            torch.any(weight <= 0.0).item()
        ):
            log_and_raise("distance weights must be finite and positive")
        if informed_slots < 1:
            log_and_raise("informed_slots must be positive")
        if informed_start < 0:
            log_and_raise("informed_start must be non-negative")
        self.register_buffer("distance_weight", weight)
        self.register_buffer("action_lows", lows)
        self.register_buffer("action_highs", highs)
        self.informed_slots = informed_slots
        self.informed_start = informed_start
        self.action_dim = weight.numel()

    def forward(
        self,
        uniform_fallback: torch.Tensor,
        unit_ball: torch.Tensor,
        starts: torch.Tensor,
        goals: torch.Tensor,
        c_max: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mixed samples and the informed slots accepted inside bounds."""
        informed_end = self.informed_start + self.informed_slots
        informed_ball = unit_ball[:, : self.informed_slots]
        fallback = uniform_fallback[:, self.informed_start : informed_end]

        weighted_starts = starts * self.distance_weight
        weighted_goals = goals * self.distance_weight
        focal_delta = weighted_goals - weighted_starts
        c_min = torch.linalg.vector_norm(focal_delta, dim=-1)
        radius_valid = torch.isfinite(c_max) & (c_max >= c_min)
        safe_c_max = torch.where(radius_valid, c_max, c_min)

        semi_major = 0.5 * safe_c_max
        semi_minor = 0.5 * torch.sqrt(torch.clamp(safe_c_max.square() - c_min.square(), min=0.0))
        scaled_ball = torch.cat(
            (
                informed_ball[..., :1] * semi_major[:, None, None],
                informed_ball[..., 1:] * semi_minor[:, None, None],
            ),
            dim=-1,
        )

        first_axis = F.one_hot(
            torch.zeros((), device=starts.device, dtype=torch.int64),
            num_classes=self.action_dim,
        ).to(dtype=starts.dtype)
        focal_direction = torch.where(
            (c_min > 1.0e-12).unsqueeze(-1),
            focal_delta / c_min.clamp_min(1.0e-12).unsqueeze(-1),
            first_axis,
        )
        reflection = first_axis - focal_direction
        reflection_norm = torch.linalg.vector_norm(reflection, dim=-1, keepdim=True)
        reflection = torch.where(
            reflection_norm > 1.0e-12,
            reflection / reflection_norm.clamp_min(1.0e-12),
            torch.zeros_like(reflection),
        )
        rotated = scaled_ball - 2.0 * reflection[:, None, :] * torch.sum(
            scaled_ball * reflection[:, None, :], dim=-1, keepdim=True
        )

        weighted_center = 0.5 * (weighted_starts + weighted_goals)
        informed = (weighted_center[:, None, :] + rotated) / self.distance_weight
        finite = torch.isfinite(informed).all(dim=-1)
        within_bounds = ((informed >= self.action_lows) & (informed <= self.action_highs)).all(
            dim=-1
        )
        accepted = radius_valid.unsqueeze(-1) & finite & within_bounds
        informed_or_fallback = torch.where(accepted.unsqueeze(-1), informed, fallback)
        mixed = torch.cat(
            (
                uniform_fallback[:, : self.informed_start],
                informed_or_fallback,
                uniform_fallback[:, informed_end:],
            ),
            dim=1,
        )
        return mixed, accepted


class InformedRRTProposalTransform(nn.Module):
    """Fuse unit-ball generation and mixed ellipsoid transformation."""

    def __init__(
        self,
        distance_weight: torch.Tensor,
        action_lows: torch.Tensor,
        action_highs: torch.Tensor,
        informed_slots: int,
        informed_start: int = 0,
    ) -> None:
        """Create one exportable robot-specific informed-sampling operation."""
        super().__init__()
        self.unit_ball = UnitBallRRTTransform(distance_weight.numel())
        self.ellipsoid = InformedEllipsoidRRTTransform(
            distance_weight,
            action_lows,
            action_highs,
            informed_slots,
            informed_start,
        )

    def forward(
        self,
        uniform_fallback: torch.Tensor,
        unit_cube: torch.Tensor,
        starts: torch.Tensor,
        goals: torch.Tensor,
        c_max: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a fixed-width proposal batch and its accepted informed-slot mask."""
        return self.ellipsoid(
            uniform_fallback,
            self.unit_ball(unit_cube),
            starts,
            goals,
            c_max,
        )


class InformedRRTProposalGenerator(nn.Module):
    """Fuse raw unit-cube decoding, radius scheduling, and informed sampling."""

    def __init__(
        self,
        distance_weight: torch.Tensor,
        action_lows: torch.Tensor,
        action_highs: torch.Tensor,
        total_slots: int,
        informed_slots: int,
        informed_start: int = 0,
        min_slack: float = 0.0,
    ) -> None:
        """Create one launchable raw-sample-to-proposal-batch transform."""
        super().__init__()
        if total_slots < 1:
            log_and_raise("total_slots must be positive")
        if informed_start + informed_slots > total_slots:
            log_and_raise("informed slots must fit inside the proposal batch")
        if not math.isfinite(min_slack) or min_slack < 0.0:
            log_and_raise("min_slack must be finite and non-negative")
        lows = action_lows.detach().clone().reshape(-1)
        highs = action_highs.detach().clone().reshape(-1)
        self.register_buffer("action_lows", lows)
        self.register_buffer("action_range", highs - lows)
        self.proposal_transform = InformedRRTProposalTransform(
            distance_weight,
            action_lows,
            action_highs,
            informed_slots,
            informed_start,
        )
        self.total_slots = total_slots
        self.informed_slots = informed_slots
        self.action_dim = distance_weight.numel()
        self.min_slack = min_slack

    def forward(
        self,
        raw_uniform: torch.Tensor,
        starts: torch.Tensor,
        goals: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform one fixed raw random stream into a mixed proposal batch."""
        uniform_fallback = (
            raw_uniform[:, : self.total_slots] * self.action_range + self.action_lows
        )
        latent_values = self.informed_slots * (self.action_dim + 1)
        unit_cube = raw_uniform[:, self.total_slots :].reshape(raw_uniform.shape[0], -1)
        unit_cube = unit_cube[:, :latent_values].reshape(
            raw_uniform.shape[0],
            self.informed_slots,
            self.action_dim + 1,
        )
        c_min = torch.linalg.vector_norm(
            (goals - starts) * self.proposal_transform.ellipsoid.distance_weight,
            dim=-1,
        )
        c_max = torch.maximum(scale * c_min, c_min + self.min_slack)
        return self.proposal_transform(uniform_fallback, unit_cube, starts, goals, c_max)
