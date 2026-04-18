# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

# Third Party
import numpy as np
import torch
import torch.autograd.profiler as profiler
from torch.profiler import record_function

# CuRobo
from curobo._src.geom.quaternion import (
    angular_distance_axis_angle,
    angular_distance_phi3,
    normalize_quaternion,
)
from curobo._src.geom.transform import (
    batch_transform_points,
    batch_transform_points_inverse,
    matrix_to_quaternion,
    pose_inverse,
    pose_multiply,
    pose_to_affine_matrix,
    pose_to_matrix,
    quaternion_to_matrix,
    transform_points,
)
from curobo._src.curobolib.cuda_ops.tensor_checks import (
    check_float16_tensors,
    check_float32_tensors,
)
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tensor import T_BPosition, T_BQuaternion, T_BRotation
from curobo._src.util.logging import deprecated, log_and_raise
from curobo._src.util.tensor_util import clone_if_not_none, copy_tensor


def _check_pose_optional_float_tensors(
    *,
    position: Optional[torch.Tensor] = None,
    quaternion: Optional[torch.Tensor] = None,
) -> None:
    ref = position if position is not None else quaternion
    if ref is None:
        return
    check = (
        check_float16_tensors if ref.dtype == torch.float16 else check_float32_tensors
    )
    kwargs = {}
    if position is not None:
        kwargs["position"] = position
    if quaternion is not None:
        kwargs["quaternion"] = quaternion
    check(ref.device, **kwargs)


@dataclass
class Pose(Sequence):
    """Pose representation used in CuRobo. You can initialize a pose by calling
    pose = Pose(position, quaternion).
    """

    #: Position is represented as x, y, z, in meters
    position: Optional[T_BPosition] = None
    #: Quaternion is represented as w, x, y, z.
    quaternion: Optional[T_BQuaternion] = None
    #: Rotation is represents orientation as a 3x3 rotation matrix
    rotation: Optional[T_BRotation] = None
    #: Batch size will be initialized from input.
    batch_size: int = 1
    #: name of link that this pose represents.
    name: str = "ee_link"
    #: quaternion input will be normalized when this flag is enabled. This is recommended when
    #: a pose comes from an external source as some programs do not send normalized quaternions.
    normalize_rotation: bool = False

    def __post_init__(self):
        if self.rotation is not None and self.quaternion is None:
            self.quaternion = matrix_to_quaternion(self.rotation)
        if self.position is not None:
            if len(self.position.shape) > 1:
                self.batch_size = self.position.shape[0]
            else:
                self.position = self.position.unsqueeze(0)
                self.quaternion = self.quaternion.unsqueeze(0)
        if self.quaternion is None and self.position is not None:
            q_size = list(self.position.shape)
            q_size[-1] = 4
            self.quaternion = torch.zeros(
                q_size, device=self.position.device, dtype=self.position.dtype
            )
            self.quaternion[..., 0] = 1.0
        if self.quaternion is not None and self.normalize_rotation:
            self.quaternion = normalize_quaternion(self.quaternion)

    def __eq__(self, other: Pose) -> bool:
        # check if shapes are the same
        if (
            self.position.shape != other.position.shape
            or self.quaternion.shape != other.quaternion.shape
        ):
            print("shape mismatch: ", self.position.shape, other.position.shape)
            return False
        p_distance, q_distance = self.distance(other)

        if torch.max(p_distance) > 1e-6:
            return False
        if torch.max(q_distance) > 1e-6:
            return False
        return True

    def detach(self):
        return Pose(
            position=self.position.detach() if self.position is not None else None,
            quaternion=self.quaternion.detach() if self.quaternion is not None else None,
            rotation=self.rotation.detach() if self.rotation is not None else None,
            normalize_rotation=False,
        )

    def requires_grad_(self, requires_grad: bool):
        if self.position is not None:
            self.position.requires_grad_(requires_grad)
        if self.quaternion is not None:
            self.quaternion.requires_grad_(requires_grad)

    def _update_shape_params(self):
        if len(self.position.shape) > 1:
            self.batch_size = self.position.shape[0]

    @property
    @deprecated(reason="Use batch_size instead")
    def batch(self):
        return self.batch_size

    @property
    def device(self):
        return self.position.device

    @property
    def ndim(self):
        return self.position.ndim

    @staticmethod
    def from_matrix(matrix: Union[np.ndarray, torch.Tensor]):
        if not isinstance(matrix, torch.Tensor):
            device_cfg = DeviceCfg()
            matrix = torch.as_tensor(matrix, device=device_cfg.device, dtype=device_cfg.dtype)

        if len(matrix.shape) == 2:
            matrix = matrix.view(-1, 4, 4)
        return Pose(
            position=matrix[..., :3, 3].contiguous(),
            rotation=matrix[..., :3, :3].contiguous(),
            normalize_rotation=True,
        )

    @staticmethod
    def _euler_xyz_to_quaternion(euler_xyz: torch.Tensor) -> torch.Tensor:
        """Convert XYZ Euler angles to quaternion (wxyz format).

        This uses the extrinsic XYZ convention (equivalent to intrinsic ZYX):
        rotations are applied around fixed world axes in order X, Y, Z.

        Args:
            euler_xyz: [..., 3] Euler angles [rx, ry, rz] in radians.

        Returns:
            [..., 4] quaternion in wxyz format.
        """
        # Half angles
        rx = euler_xyz[..., 0:1] * 0.5
        ry = euler_xyz[..., 1:2] * 0.5
        rz = euler_xyz[..., 2:3] * 0.5

        cx = torch.cos(rx)
        sx = torch.sin(rx)
        cy = torch.cos(ry)
        sy = torch.sin(ry)
        cz = torch.cos(rz)
        sz = torch.sin(rz)

        # Quaternion from extrinsic XYZ (= intrinsic ZYX): q = qz * qy * qx
        # qx = [cx, sx, 0, 0], qy = [cy, 0, sy, 0], qz = [cz, 0, 0, sz]
        w = cx * cy * cz + sx * sy * sz
        x = sx * cy * cz - cx * sy * sz
        y = cx * sy * cz + sx * cy * sz
        z = cx * cy * sz - sx * sy * cz

        return torch.cat([w, x, y, z], dim=-1)

    @classmethod
    def from_euler_xyz(
        cls,
        euler_xyz: torch.Tensor,
        position: Optional[torch.Tensor] = None,
    ) -> "Pose":
        """Create a Pose from XYZ Euler angles and optional position.

        This uses the extrinsic XYZ convention: rotations are applied around
        fixed world axes in order X, Y, Z.

        Args:
            euler_xyz: [..., 3] Euler angles [rx, ry, rz] in radians.
            position: Optional [..., 3] position. If None, defaults to zero.

        Returns:
            Pose with the specified position (or zero) and the corresponding quaternion.
        """
        quaternion = cls._euler_xyz_to_quaternion(euler_xyz)
        if quaternion.ndim == 1:
            quaternion = quaternion.unsqueeze(0)
        if position is None:
            position = torch.zeros(
                quaternion.shape[:-1] + (3,),
                device=euler_xyz.device,
                dtype=euler_xyz.dtype,
            )
        elif position.ndim == 1:
            position = position.unsqueeze(0)
        return cls(position=position, quaternion=quaternion, normalize_rotation=False)

    @staticmethod
    def _euler_xyz_intrinsic_to_quaternion(euler_xyz: torch.Tensor) -> torch.Tensor:
        """Convert XYZ Euler angles to quaternion (wxyz) using intrinsic convention.

        Intrinsic XYZ (equivalent to extrinsic ZYX): each rotation is applied
        around the current body-frame axis. This matches URDF joint chains
        where successive revolute joints rotate in the parent frame:
        X (roll) → Y (pitch) → Z (yaw), giving q = q_x * q_y * q_z.

        Args:
            euler_xyz: [..., 3] Euler angles [rx, ry, rz] in radians.

        Returns:
            [..., 4] quaternion in wxyz format.
        """
        rx = euler_xyz[..., 0:1] * 0.5
        ry = euler_xyz[..., 1:2] * 0.5
        rz = euler_xyz[..., 2:3] * 0.5

        cx = torch.cos(rx)
        sx = torch.sin(rx)
        cy = torch.cos(ry)
        sy = torch.sin(ry)
        cz = torch.cos(rz)
        sz = torch.sin(rz)

        w = cx * cy * cz - sx * sy * sz
        x = sx * cy * cz + cx * sy * sz
        y = cx * sy * cz - sx * cy * sz
        z = cx * cy * sz + sx * sy * cz

        return torch.cat([w, x, y, z], dim=-1)

    @classmethod
    def from_euler_xyz_intrinsic(
        cls,
        euler_xyz: torch.Tensor,
        position: Optional[torch.Tensor] = None,
    ) -> "Pose":
        """Create a Pose from XYZ Euler angles using intrinsic convention.

        Intrinsic XYZ: each rotation is applied around the current body-frame
        axis (X → Y → Z). This matches URDF virtual base joint chains where
        successive revolute joints rotate in the parent frame.

        See :meth:`from_euler_xyz` for the extrinsic convention.

        Args:
            euler_xyz: [..., 3] Euler angles [rx, ry, rz] in radians.
            position: Optional [..., 3] position. If None, defaults to zero.

        Returns:
            Pose with the specified position (or zero) and the corresponding quaternion.
        """
        quaternion = cls._euler_xyz_intrinsic_to_quaternion(euler_xyz)
        if quaternion.ndim == 1:
            quaternion = quaternion.unsqueeze(0)
        if position is None:
            position = torch.zeros(
                quaternion.shape[:-1] + (3,),
                device=euler_xyz.device,
                dtype=euler_xyz.dtype,
            )
        elif position.ndim == 1:
            position = position.unsqueeze(0)
        return cls(position=position, quaternion=quaternion, normalize_rotation=False)

    def get_rotation_matrix(self):
        return self.get_rotation()

    def get_rotation(self):
        if self.rotation is not None:
            return self.rotation
        elif self.quaternion is not None:
            return quaternion_to_matrix(self.quaternion)
        else:
            return None

    def stack(self, other_pose: Pose):
        position = torch.vstack((self.position, other_pose.position))
        quaternion = None
        rotation = None
        if self.quaternion is not None and other_pose.quaternion is not None:
            quaternion = torch.vstack((self.quaternion, other_pose.quaternion))
        return Pose(position=position, quaternion=quaternion, rotation=rotation)

    def repeat(self, n):
        """Repeat pose

        Args:
            n ([type]): [description]
        """
        if n <= 1:
            return self

        position = self.position
        quaternion = self.quaternion

        position = self.position.repeat(n, 1)
        quaternion = self.quaternion.repeat(n, 1)
        return Pose(position=position, quaternion=quaternion, batch_size=position.shape[0])

    def unsqueeze(self, dim=-1):
        if self.position is not None:
            self.position = self.position.unsqueeze(dim)
        if self.quaternion is not None:
            self.quaternion = self.quaternion.unsqueeze(dim)
        if self.rotation is not None:
            self.rotation = self.rotation.unsqueeze(dim)
        self._update_shape_params()
        return self

    def squeeze(self, dim=-1):
        if self.position is not None:
            self.position = self.position.squeeze(dim)
        if self.quaternion is not None:
            self.quaternion = self.quaternion.squeeze(dim)
        if self.rotation is not None:
            self.rotation = self.rotation.squeeze(dim)
        return self

    def repeat_seeds(self, num_seeds: int):
        if (self.position is None and self.quaternion is None) or num_seeds <= 1:
            return Pose(self.position, self.quaternion)
        position = (
            self.position.view(self.batch_size, 1, 3)
            .repeat(1, num_seeds, 1)
            .reshape(self.batch_size * num_seeds, 3)
        )
        quaternion = (
            self.quaternion.view(self.batch_size, 1, 4)
            .repeat(1, num_seeds, 1)
            .reshape(self.batch_size * num_seeds, 4)
        )
        return Pose(position=position, quaternion=quaternion)

    def __getitem__(self, idx: Union[int, torch.Tensor]):
        # idx_vals = list_idx_if_not_none([self.position, self.quaternion], idx)

        return Pose(
            position=self.position[idx], quaternion=self.quaternion[idx], normalize_rotation=False
        )

    def __setitem__(self, idx: Union[int, torch.Tensor], value: Pose):
        # idx_vals = list_idx_if_not_none([self.position, self.quaternion], idx)
        self.position[idx] = value.position
        self.quaternion[idx] = value.quaternion

    def __len__(self):
        return self.batch_size

    def get_index(self, b: int, n: Optional[int] = None) -> Pose:
        if n is None:
            position = self.position[b, :]
            quaternion = self.quaternion[b, :]
        else:
            position = self.position[b, n, :]
            quaternion = self.quaternion[b, n, :]
        _check_pose_optional_float_tensors(position=position, quaternion=quaternion)
        return Pose(
            position=position,
            quaternion=quaternion,
            normalize_rotation=False,
        )

    def apply_kernel(self, kernel_mat):
        if self.position is None:
            return self
        return Pose(
            position=kernel_mat @ self.position, quaternion=kernel_mat @ self.quaternion
        )

    @classmethod
    def from_numpy(
        cls,
        position: np.ndarray,
        quaternion: np.ndarray,
        device_cfg: DeviceCfg = DeviceCfg(),
    ):
        position = torch.as_tensor(position, device=device_cfg.device, dtype=device_cfg.dtype)
        quaternion = torch.as_tensor(quaternion, device=device_cfg.device, dtype=device_cfg.dtype)
        return cls(position=position, quaternion=quaternion)

    @classmethod
    def from_list(
        cls, pose: List[float], device_cfg: DeviceCfg = DeviceCfg(), q_xyzw=False
    ):
        position = torch.as_tensor(
            pose[:3], device=device_cfg.device, dtype=device_cfg.dtype
        ).unsqueeze(0)
        if q_xyzw:
            quaternion = torch.as_tensor(
                [pose[6], pose[3], pose[4], pose[5]],
                device=device_cfg.device,
                dtype=device_cfg.dtype,
            ).unsqueeze(0)
        else:
            quaternion = torch.as_tensor(
                pose[3:], device=device_cfg.device, dtype=device_cfg.dtype
            ).unsqueeze(0)
        return Pose(position=position, quaternion=quaternion)

    @classmethod
    def from_batch_list(
        cls,
        pose: List[List[float]],
        device_cfg: DeviceCfg = DeviceCfg(),
        q_xyzw=False,
    ):
        # create a cpu pytorch tensor first
        pose_mat = np.array(pose)
        position = torch.as_tensor(
            pose_mat[..., :3], device=device_cfg.device, dtype=device_cfg.dtype
        )
        quaternion = torch.as_tensor(
            pose_mat[..., 3:], device=device_cfg.device, dtype=device_cfg.dtype
        )
        if q_xyzw:
            q_c = quaternion.clone()
            quaternion[..., 0] = q_c[..., 3]
            quaternion[..., 1] = q_c[..., 0]
            quaternion[..., 2] = q_c[..., 1]
            quaternion[..., 3] = q_c[..., 2]

        return Pose(position=position, quaternion=quaternion)

    def to_list(self, q_xyzw=False):
        return self.tolist(q_xyzw)

    def tolist(self, q_xyzw=False):
        if q_xyzw:
            q = self.quaternion.cpu().squeeze().tolist()
            return self.position.cpu().squeeze().tolist() + [q[1], q[2], q[3], q[0]]

        return self.position.cpu().squeeze().tolist() + self.quaternion.cpu().squeeze().tolist()

    def clone(self):
        return Pose(
            position=clone_if_not_none(self.position),
            quaternion=clone_if_not_none(self.quaternion),
            normalize_rotation=False,
        )

    def to(
        self,
        device_cfg: Optional[DeviceCfg] = None,
        device: Optional[torch.device] = None,
    ):
        if device_cfg is None and device is None:
            log_and_raise("Pose.to() requires device_cfg or device")
        if device_cfg is not None:
            t_type = device_cfg.as_torch_dict()
        else:
            t_type = {"device": device}
        if self.position is not None:
            self.position = self.position.to(**t_type)
        if self.quaternion is not None:
            self.quaternion = self.quaternion.to(**t_type)
        if self.rotation is not None:
            self.rotation = self.rotation.to(**t_type)
        return self

    @profiler.record_function("pose/get_matrix")
    def get_matrix(self, out_matrix: Optional[torch.Tensor] = None):
        full_mat = pose_to_matrix(self.position, self.quaternion, out_matrix)
        return full_mat

    def get_affine_matrix(self, out_matrix: Optional[torch.Tensor] = None):
        full_mat = pose_to_affine_matrix(self.position, self.quaternion, out_matrix)
        return full_mat

    def get_numpy_affine_matrix(self):
        return self.get_affine_matrix().cpu().numpy()

    def get_numpy_matrix(self):
        return self.get_matrix().cpu().numpy()

    @profiler.record_function("pose/inverse")
    def inverse(self):
        """Inverse of pose

        Returns:
            Pose: inverse pose
        """
        # rot, position = get_inv_transform(self.get_rotation(), self.position)
        # out = Pose(position, rotation=rot)
        # return out

        position, quaternion = pose_inverse(self.position, self.quaternion)
        out = Pose(position.clone(), quaternion.clone())
        return out

    def get_pose_vector(self):
        return torch.cat((self.position, self.quaternion), dim=-1)

    def copy_(self, pose: Pose):
        """Copies pose data from another memory buffer.
        This will create a new instance if buffers are not same shape

        Args:
            pose (Pose): _description_
        """
        if pose.position is None and pose.quaternion is None:
            log_and_raise("Pose.copy_(): pose.position and pose.quaternion are None")
        if not copy_tensor(pose.position, self.position) or not copy_tensor(
            pose.quaternion, self.quaternion
        ):
            log_and_raise(
                f"Copy not possible due to shape mismatch: {pose.position.shape} != {self.position.shape}"
            )
        return

    @staticmethod
    def cat(pose_list: List[Pose]):
        position_cat = torch.cat([i.position for i in pose_list])
        quaternion_cat = torch.cat([i.quaternion for i in pose_list])
        return Pose(position=position_cat, quaternion=quaternion_cat, normalize_rotation=False)

    def distance(self, other_pose: Pose, use_phi3: bool = False):
        quat_distance = self.angular_distance(other_pose, use_phi3)
        p_distance = self.linear_distance(other_pose)
        return p_distance, quat_distance

    def angular_distance(self, other_pose: Pose, use_phi3: bool = False):
        """This function computes the angular distance using either phi_3 or axis-angle.

        See Huynh, Du Q. "Metrics for 3D rotations: Comparison and analysis." Journal of
        Mathematical Imaging and Vision 35 (2009): 155-164 for phi_3 metric.

        Args:
            current_quat: other pose quaternion. Shape: [..., 4]

        Returns:
            Angular distance in range [0,1]. Shape: [...]
        """
        if use_phi3:
            quat_distance = angular_distance_phi3(self.quaternion, other_pose.quaternion)

        else:
            quat_distance = angular_distance_axis_angle(self.quaternion, other_pose.quaternion)
        return quat_distance

    def linear_distance(self, other_pose: Pose):
        p_distance = torch.linalg.norm(self.position - other_pose.position, dim=-1)
        return p_distance

    @profiler.record_function("pose/multiply")
    def multiply(self, other_pose: Pose, out_position: Optional[torch.Tensor] = None, out_quaternion: Optional[torch.Tensor] = None):
        if self.shape == other_pose.shape or (
            (self.shape[0] == 1 and other_pose.shape[0] > 1) and len(other_pose.shape) == 2
        ):
            p3, q3 = pose_multiply(
                self.position, self.quaternion, other_pose.position, other_pose.quaternion, out_position, out_quaternion
            )
            return Pose(p3, q3)
        else:
            mat_mul = self.get_matrix() @ other_pose.get_matrix()
            return Pose.from_matrix(mat_mul)

    @deprecated(reason="Use Pose.transform_points instead")
    def transform_point(
        self,
        points: torch.Tensor,
        out_buffer: Optional[torch.Tensor] = None,
        gp_out: Optional[torch.Tensor] = None,
        gq_out: Optional[torch.Tensor] = None,
        gpt_out: Optional[torch.Tensor] = None,
    ):
        if len(points.shape) > 2:
            points = points.view(-1, 3)
        return transform_points(
            self.position, self.quaternion, points, out_buffer, gp_out, gq_out, gpt_out
        )

    def transform_points(
        self,
        points: torch.Tensor,
        out_buffer: Optional[torch.Tensor] = None,
        gp_out: Optional[torch.Tensor] = None,
        gq_out: Optional[torch.Tensor] = None,
        gpt_out: Optional[torch.Tensor] = None,
    ):
        if len(points.shape) > 2:
            points = points.view(-1, 3)
        return transform_points(
            self.position.view(-1, 3),
            self.quaternion.view(-1, 4),
            points,
            out_buffer,
            gp_out,
            gq_out,
            gpt_out,
        )

    @record_function("math/pose/transform_points")
    def batch_transform_points(
        self,
        points: torch.Tensor,
        out_buffer: Optional[torch.Tensor] = None,
        gp_out: Optional[torch.Tensor] = None,
        gq_out: Optional[torch.Tensor] = None,
        gpt_out: Optional[torch.Tensor] = None,
    ):
        if len(points.shape) <= 2:
            log_and_raise("batch_transform requires points to be b,n,3 shape")
        return batch_transform_points(
            self.position.view(-1, 3),
            self.quaternion.view(-1, 4),
            points,
            out_buffer,
            gp_out,
            gq_out,
            gpt_out,
        )

    def batch_transform_points_inverse(self, points: torch.Tensor, out_buffer: Optional[torch.Tensor] = None, gp_out: Optional[torch.Tensor] = None, gq_out: Optional[torch.Tensor] = None, gpt_out: Optional[torch.Tensor] = None):
        return batch_transform_points_inverse(
            self.position.view(-1, 3),
            self.quaternion.view(-1, 4),
            points,
            out_buffer,
            gp_out,
            gq_out,
            gpt_out,
        )
    @property
    def shape(self):
        return self.position.shape

    def compute_offset_pose(self, offset: Pose) -> Pose:
        return self.multiply(offset)

    def compute_local_pose(self, world_pose: Pose) -> Pose:
        return self.inverse().multiply(world_pose)

    def contiguous(self) -> Pose:
        _check_pose_optional_float_tensors(
            position=self.position, quaternion=self.quaternion
        )
        return Pose(
            position=self.position,
            quaternion=self.quaternion,
            normalize_rotation=False,
        )
