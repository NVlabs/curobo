# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query targets for transforms relative to one moving tool frame."""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import List

# Third Party
import torch

# CuRobo
from curobo._src.geom.quaternion import quat_multiply
from curobo._src.geom.transform import torch_quaternion_to_matrix
from curobo._src.types.tool_pose import ToolPose
from curobo._src.util.logging import log_and_raise


def _normalize_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Normalize a validated nonzero quaternion and select a stable sign."""
    normalized = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    return torch.where(normalized[..., :1] < 0.0, -normalized, normalized)


@dataclass
class ToolRelativeTransformTarget:
    """Target transforms from one moving reference tool frame to each tool.

    Each entry is ``T_reference_tool = inverse(T_world_reference) * T_world_tool``.
    Quaternion values are not implicitly normalized, so scaled nonzero
    quaternions retain the caller's storage while representing the same rotation.
    """

    #: Tool frame in which every target transform is expressed.
    reference_tool_frame: str

    #: Ordered tool-frame names matching dimension 1 of :attr:`position` and
    #: :attr:`quaternion`.
    tool_frames: List[str]

    #: Translation of ``T_reference_tool`` with shape
    #: ``[num_target_groups, num_tool_frames, 3]``.
    position: torch.Tensor

    #: Rotation of ``T_reference_tool`` with shape
    #: ``[num_target_groups, num_tool_frames, 4]`` in wxyz convention.
    quaternion: torch.Tensor

    def __post_init__(self) -> None:
        """Validate tensor layout and self-describing frame metadata."""
        if self.position.ndim != 3:
            log_and_raise(
                "ToolRelativeTransformTarget position must be 3D "
                "[num_target_groups, num_tool_frames, 3], "
                f"got {self.position.shape}"
            )
        if self.quaternion.ndim != 3:
            log_and_raise(
                "ToolRelativeTransformTarget quaternion must be 3D "
                "[num_target_groups, num_tool_frames, 4], "
                f"got {self.quaternion.shape}"
            )
        if self.position.shape[-1] != 3:
            log_and_raise(
                "ToolRelativeTransformTarget position last dimension must be 3, "
                f"got {self.position.shape[-1]}"
            )
        if self.quaternion.shape[-1] != 4:
            log_and_raise(
                "ToolRelativeTransformTarget quaternion last dimension must be 4, "
                f"got {self.quaternion.shape[-1]}"
            )
        if self.position.shape[:2] != self.quaternion.shape[:2]:
            log_and_raise(
                "ToolRelativeTransformTarget position and quaternion leading dimensions "
                f"must match, got {self.position.shape[:2]} and {self.quaternion.shape[:2]}"
            )
        if self.position.shape[0] < 1:
            log_and_raise("ToolRelativeTransformTarget requires at least one target group")
        if self.position.shape[1] != len(self.tool_frames):
            log_and_raise(
                f"num_tool_frames dimension ({self.position.shape[1]}) does not match "
                f"len(tool_frames) ({len(self.tool_frames)})"
            )
        if len(set(self.tool_frames)) != len(self.tool_frames):
            log_and_raise("ToolRelativeTransformTarget tool_frames must be unique")
        if self.reference_tool_frame not in self.tool_frames:
            log_and_raise(
                "ToolRelativeTransformTarget reference_tool_frame "
                f"'{self.reference_tool_frame}' is not in tool_frames {self.tool_frames}"
            )
        if self.position.device != self.quaternion.device:
            log_and_raise(
                "ToolRelativeTransformTarget position and quaternion must be on the same device"
            )
        if self.position.dtype != self.quaternion.dtype:
            log_and_raise(
                "ToolRelativeTransformTarget position and quaternion must have the same dtype"
            )
        if not self.position.dtype.is_floating_point:
            log_and_raise(
                "ToolRelativeTransformTarget position and quaternion must use a floating dtype"
            )
        if not bool(torch.isfinite(self.position).all()):
            log_and_raise("ToolRelativeTransformTarget position must contain only finite values")
        if not bool(torch.isfinite(self.quaternion).all()):
            log_and_raise("ToolRelativeTransformTarget quaternion must contain only finite values")
        quaternion_norm = torch.linalg.vector_norm(self.quaternion, dim=-1)
        if bool(torch.any(quaternion_norm <= torch.finfo(self.quaternion.dtype).eps)):
            log_and_raise("ToolRelativeTransformTarget quaternion norm must be nonzero")

    @classmethod
    def from_tool_pose(
        cls,
        tool_pose: ToolPose,
        reference_tool_frame: str,
        horizon_index: int = 0,
    ) -> ToolRelativeTransformTarget:
        """Create reference-relative targets from one FK horizon index.

        The FK batch dimension becomes the target-group dimension. For every
        tool frame ``B``, this computes
        ``T_reference_B = inverse(T_world_reference) * T_world_B`` while
        preserving the tool-frame order from ``tool_pose``. Input and output
        quaternions use CuRobo's ``wxyz`` convention; output quaternions are
        normalized and canonicalized to have a nonnegative scalar component.

        Args:
            tool_pose: Batched FK output with shape ``[B, H, L, 3/4]``.
            reference_tool_frame: Tool frame in which to express every target.
            horizon_index: FK horizon index to capture. Defaults to ``0``.

        Returns:
            Relative targets with shape ``[B, L, 3/4]``.
        """
        if not isinstance(tool_pose, ToolPose):
            log_and_raise("ToolRelativeTransformTarget.from_tool_pose requires a ToolPose input")
        if isinstance(horizon_index, bool) or not isinstance(horizon_index, int):
            log_and_raise("horizon_index must be an integer")
        if reference_tool_frame not in tool_pose.tool_frames:
            log_and_raise(
                "reference_tool_frame "
                f"'{reference_tool_frame}' is not in tool_frames {tool_pose.tool_frames}"
            )

        position = tool_pose.position
        quaternion = tool_pose.quaternion
        if position.shape[-1] != 3:
            log_and_raise(f"ToolPose position last dimension must be 3, got {position.shape[-1]}")
        if quaternion.shape[-1] != 4:
            log_and_raise(
                f"ToolPose quaternion last dimension must be 4, got {quaternion.shape[-1]}"
            )
        if position.shape[:3] != quaternion.shape[:3]:
            log_and_raise(
                "ToolPose position and quaternion leading dimensions must match, "
                f"got {position.shape[:3]} and {quaternion.shape[:3]}"
            )
        if position.shape[2] != len(tool_pose.tool_frames):
            log_and_raise(
                f"ToolPose num_links dimension ({position.shape[2]}) does not match "
                f"len(tool_frames) ({len(tool_pose.tool_frames)})"
            )
        if position.shape[0] < 1:
            log_and_raise("ToolPose requires at least one batch element")
        if position.shape[1] < 1:
            log_and_raise("ToolPose requires at least one horizon element")
        if horizon_index < 0 or horizon_index >= position.shape[1]:
            log_and_raise(
                f"horizon_index {horizon_index} is out of range for horizon {position.shape[1]}"
            )
        if position.device != quaternion.device:
            log_and_raise("ToolPose position and quaternion must be on the same device")
        if position.dtype != quaternion.dtype:
            log_and_raise("ToolPose position and quaternion must have the same dtype")
        if not position.dtype.is_floating_point:
            log_and_raise("ToolPose position and quaternion must use a floating dtype")

        world_position = position[:, horizon_index]
        world_quaternion_input = quaternion[:, horizon_index]
        if not bool(torch.isfinite(world_position).all()):
            log_and_raise("ToolPose position must contain only finite values")
        if not bool(torch.isfinite(world_quaternion_input).all()):
            log_and_raise("ToolPose quaternion must contain only finite values")
        quaternion_norm = torch.linalg.vector_norm(world_quaternion_input, dim=-1)
        if bool(torch.any(quaternion_norm <= torch.finfo(quaternion.dtype).eps)):
            log_and_raise("ToolPose quaternion norm must be nonzero")

        world_quaternion = _normalize_wxyz(world_quaternion_input)
        reference_index = tool_pose.tool_frames.index(reference_tool_frame)
        reference_position = world_position[:, reference_index : reference_index + 1]
        reference_quaternion = world_quaternion[:, reference_index : reference_index + 1]

        reference_rotation = torch_quaternion_to_matrix(reference_quaternion[:, 0])
        relative_position = torch.matmul(
            reference_rotation.transpose(-1, -2).unsqueeze(1),
            (world_position - reference_position).unsqueeze(-1),
        ).squeeze(-1)

        inverse_reference_quaternion = torch.cat(
            (reference_quaternion[..., :1], -reference_quaternion[..., 1:]),
            dim=-1,
        )
        relative_quaternion = quat_multiply(
            inverse_reference_quaternion.expand_as(world_quaternion),
            world_quaternion,
        )
        relative_quaternion = _normalize_wxyz(relative_quaternion)

        return cls(
            reference_tool_frame=reference_tool_frame,
            tool_frames=tool_pose.tool_frames.copy(),
            position=relative_position.contiguous(),
            quaternion=relative_quaternion.contiguous(),
        )

    @property
    def num_target_groups(self) -> int:
        """Return the number of target transform groups."""
        return self.position.shape[0]

    @property
    def num_tool_frames(self) -> int:
        """Return the number of aligned tool frames."""
        return self.position.shape[1]

    @property
    def shape(self) -> torch.Size:
        """Return the relative-position tensor shape."""
        return self.position.shape

    @property
    def ndim(self) -> int:
        """Return the relative-position tensor rank."""
        return self.position.ndim

    @property
    def device(self) -> torch.device:
        """Return the target tensor device."""
        return self.position.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the target tensor dtype."""
        return self.position.dtype

    def copy_(self, other: ToolRelativeTransformTarget) -> None:
        """Copy compatible target tensor values into the current storage."""
        if self.reference_tool_frame != other.reference_tool_frame:
            log_and_raise(
                "Cannot copy ToolRelativeTransformTarget with a different "
                f"reference_tool_frame: {self.reference_tool_frame} != "
                f"{other.reference_tool_frame}"
            )
        if self.tool_frames != other.tool_frames:
            log_and_raise(
                "Cannot copy ToolRelativeTransformTarget with different tool_frames: "
                f"{self.tool_frames} != {other.tool_frames}"
            )
        if self.position.shape != other.position.shape:
            log_and_raise(
                "Cannot copy ToolRelativeTransformTarget with a different shape: "
                f"{self.position.shape} != {other.position.shape}"
            )
        if self.device != other.device:
            log_and_raise(
                "Cannot copy ToolRelativeTransformTarget from a different device: "
                f"{self.device} != {other.device}"
            )
        if self.dtype != other.dtype:
            log_and_raise(
                "Cannot copy ToolRelativeTransformTarget with a different dtype: "
                f"{self.dtype} != {other.dtype}"
            )
        self.position.copy_(other.position)
        self.quaternion.copy_(other.quaternion)

    def clone(self) -> ToolRelativeTransformTarget:
        """Return an independent copy of the target metadata and tensors."""
        return ToolRelativeTransformTarget(
            reference_tool_frame=self.reference_tool_frame,
            tool_frames=self.tool_frames.copy(),
            position=self.position.clone(),
            quaternion=self.quaternion.clone(),
        )

    def detach(self) -> ToolRelativeTransformTarget:
        """Return a target whose tensors are detached from autograd."""
        return ToolRelativeTransformTarget(
            reference_tool_frame=self.reference_tool_frame,
            tool_frames=self.tool_frames.copy(),
            position=self.position.detach(),
            quaternion=self.quaternion.detach(),
        )

    def contiguous(self) -> ToolRelativeTransformTarget:
        """Return a target with contiguous tensor storage."""
        return ToolRelativeTransformTarget(
            reference_tool_frame=self.reference_tool_frame,
            tool_frames=self.tool_frames.copy(),
            position=self.position.contiguous(),
            quaternion=self.quaternion.contiguous(),
        )

    def requires_grad_(self, requires_grad: bool = True) -> ToolRelativeTransformTarget:
        """Set whether target tensors require gradients and return this target."""
        self.position.requires_grad_(requires_grad)
        self.quaternion.requires_grad_(requires_grad)
        return self
