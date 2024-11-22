#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

# Third Party
import numpy as np
import torch
import torch.autograd.profiler as profiler
from torch.autograd import Function
from torch.profiler import record_function

# CuRobo
from curobo.geom.transform import (
    batch_transform_points,
    matrix_to_quaternion,
    pose_inverse,
    pose_multiply,
    pose_to_matrix,
    quaternion_to_matrix,
    transform_points,
)
from curobo.types.base import TensorDeviceType
from curobo.util.helpers import list_idx_if_not_none
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util.tensor_util import clone_if_not_none, copy_tensor
from curobo.util.torch_utils import get_torch_jit_decorator

# Local Folder
from .tensor import T_BPosition, T_BQuaternion, T_BRotation


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
    batch: int = 1
    #: Goalset will be initialized from input when position shape is batch x n_goalset x 3
    n_goalset: int = 1
    #: name of link that this pose represents.
    name: str = "ee_link"
    #: quaternion input will be normalized when this flag is enabled. This is recommended when
    #: a pose comes from an external source as some programs do not send normalized quaternions.
    normalize_rotation: bool = True

    def __post_init__(self):
        if self.rotation is not None and self.quaternion is None:
            self.quaternion = matrix_to_quaternion(self.rotation)
        if self.position is not None:
            if len(self.position.shape) > 2:
                self.n_goalset = self.position.shape[1]
            if len(self.position.shape) > 1:
                self.batch = self.position.shape[0]
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

    @staticmethod
    def from_matrix(matrix: Union[np.ndarray, torch.Tensor]):
        if not isinstance(matrix, torch.Tensor):
            tensor_args = TensorDeviceType()
            matrix = torch.as_tensor(matrix, device=tensor_args.device, dtype=tensor_args.dtype)

        if len(matrix.shape) == 2:
            matrix = matrix.view(-1, 4, 4)
        return Pose(
            position=matrix[..., :3, 3], rotation=matrix[..., :3, :3], normalize_rotation=True
        )

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

        if self.n_goalset > 1:
            position = self.position.repeat(n, 1, 1)
            quaternion = self.quaternion.repeat(n, 1, 1)
        else:
            position = self.position.repeat(n, 1)
            quaternion = self.quaternion.repeat(n, 1)
        return Pose(position=position, quaternion=quaternion, batch=position.shape[0])

    def unsqueeze(self, dim=-1):
        if self.position is not None:
            self.position = self.position.unsqueeze(dim)
        if self.quaternion is not None:
            self.quaternion = self.quaternion.unsqueeze(dim)
        if self.rotation is not None:
            self.rotation = self.rotation.unsqueeze(dim)
        return self

    def repeat_seeds(self, num_seeds: int):
        if (self.position is None and self.quaternion is None) or num_seeds <= 1:
            return Pose(self.position, self.quaternion)
        if self.n_goalset <= 1:
            position = (
                self.position.view(self.batch, 1, 3)
                .repeat(1, num_seeds, 1)
                .reshape(self.batch * num_seeds, 3)
            )
            quaternion = (
                self.quaternion.view(self.batch, 1, 4)
                .repeat(1, num_seeds, 1)
                .reshape(self.batch * num_seeds, 4)
            )
        else:
            position = (
                self.position.view(self.batch, 1, self.n_goalset, 3)
                .repeat(1, num_seeds, 1, 1)
                .reshape(self.batch * num_seeds, self.n_goalset, 3)
            )
            quaternion = (
                self.quaternion.view(self.batch, 1, self.n_goalset, 4)
                .repeat(1, num_seeds, 1, 1)
                .reshape(self.batch * num_seeds, self.n_goalset, 4)
            )
        return Pose(position=position, quaternion=quaternion)

    def __getitem__(self, idx):
        idx_vals = list_idx_if_not_none([self.position, self.quaternion], idx)

        return Pose(position=idx_vals[0], quaternion=idx_vals[1], normalize_rotation=False)

    def __len__(self):
        return self.batch

    def get_index(self, b: int, n: Optional[int] = None) -> Pose:
        if n is None:
            return Pose(
                position=self.position[b, :].contiguous(),
                quaternion=self.quaternion[b, :].contiguous(),
                normalize_rotation=False,
            )
        else:
            return Pose(
                position=self.position[b, n, :].contiguous(),
                quaternion=self.quaternion[b, n, :].contiguous(),
                normalize_rotation=False,
            )

    def apply_kernel(self, kernel_mat):
        if self.position is None:
            return self
        if self.n_goalset <= 1:
            return Pose(
                position=kernel_mat @ self.position, quaternion=kernel_mat @ self.quaternion
            )
        else:
            pos = self.position.view(self.batch, -1)
            quat = self.quaternion.view(self.batch, -1)
            pos_k = kernel_mat @ pos
            quat_k = kernel_mat @ quat
            return Pose(
                position=pos_k.view(-1, self.n_goalset, 3),
                quaternion=quat_k.view(-1, self.n_goalset, 4),
                normalize_rotation=False,
            )

    @classmethod
    def from_list(
        cls, pose: List[float], tensor_args: TensorDeviceType = TensorDeviceType(), q_xyzw=False
    ):
        position = torch.as_tensor(
            pose[:3], device=tensor_args.device, dtype=tensor_args.dtype
        ).unsqueeze(0)
        if q_xyzw:
            quaternion = torch.as_tensor(
                [pose[6], pose[3], pose[4], pose[5]],
                device=tensor_args.device,
                dtype=tensor_args.dtype,
            ).unsqueeze(0)
        else:
            quaternion = torch.as_tensor(
                pose[3:], device=tensor_args.device, dtype=tensor_args.dtype
            ).unsqueeze(0)
        return Pose(position=position, quaternion=quaternion)

    @classmethod
    def from_batch_list(cls, pose: List[List[float]], tensor_args: TensorDeviceType, q_xyzw=False):
        # create a cpu pytorch tensor first
        pose_mat = np.array(pose)
        position = torch.as_tensor(
            pose_mat[..., :3], device=tensor_args.device, dtype=tensor_args.dtype
        )
        quaternion = torch.as_tensor(
            pose_mat[..., 3:], device=tensor_args.device, dtype=tensor_args.dtype
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
        tensor_args: Optional[TensorDeviceType] = None,
        device: Optional[torch.device] = None,
    ):
        if tensor_args is None and device is None:
            log_error("Pose.to() requires tensor_args or device")
        if tensor_args is not None:
            t_type = tensor_args.as_torch_dict()
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
            return self
        if not copy_tensor(pose.position, self.position) or not copy_tensor(
            pose.quaternion, self.quaternion
        ):
            log_info("Cloning math.Pose")

            position = pose.position.clone()
            quaternion = pose.quaternion.clone()
            batch = position.shape[0]
            return Pose(
                position=position,
                quaternion=quaternion,
                batch=batch,
                normalize_rotation=pose.normalize_rotation,
            )
        self.position.copy_(pose.position)
        self.quaternion.copy_(pose.quaternion)
        return self

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
        """This function computes the angular distance phi_3.

        See Huynh, Du Q. "Metrics for 3D rotations: Comparison and analysis." Journal of Mathematical
        Imaging and Vision 35 (2009): 155-164.

        Args:
            goal_quat: _description_
            current_quat: _description_

        Returns:
            Angular distance in range [0,1]
        """
        if use_phi3:
            quat_distance = angular_distance_phi3(self.quaternion, other_pose.quaternion)

        else:
            quat_distance = OrientationError.apply(
                self.quaternion, other_pose.quaternion, self.quaternion.clone()
            )
        return quat_distance

    def linear_distance(self, other_pose: Pose):
        p_distance = torch.linalg.norm(self.position - other_pose.position, dim=-1)
        return p_distance

    @profiler.record_function("pose/multiply")
    def multiply(self, other_pose: Pose):
        if self.shape == other_pose.shape or (
            (self.shape[0] == 1 and other_pose.shape[0] > 1) and len(other_pose.shape) == 2
        ):
            p3, q3 = pose_multiply(
                self.position, self.quaternion, other_pose.position, other_pose.quaternion
            )
            return Pose(p3, q3)
        else:
            mat_mul = self.get_matrix() @ other_pose.get_matrix()
            return Pose.from_matrix(mat_mul)

    def transform_point(
        self,
        points: torch.Tensor,
        out_buffer: Optional[torch.Tensor] = None,
        gp_out: Optional[torch.Tensor] = None,
        gq_out: Optional[torch.Tensor] = None,
        gpt_out: Optional[torch.Tensor] = None,
    ):
        log_warn("Deprecated, use Pose.transform_points instead")
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
            log_error("batch_transform requires points to be b,n,3 shape")
        return batch_transform_points(
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
        return Pose(
            position=self.position.contiguous() if self.position is not None else None,
            quaternion=self.quaternion.contiguous() if self.quaternion is not None else None,
            normalize_rotation=False,
        )


def quat_multiply(q1, q2, q_res):
    a_w = q1[..., 0]
    a_x = q1[..., 1]
    a_y = q1[..., 2]
    a_z = q1[..., 3]
    b_w = q2[..., 0]
    b_x = q2[..., 1]
    b_y = q2[..., 2]
    b_z = q2[..., 3]

    q_res[..., 0] = a_w * b_w - a_x * b_x - a_y * b_y - a_z * b_z

    q_res[..., 1] = a_w * b_x + b_w * a_x + a_y * b_z - b_y * a_z
    q_res[..., 2] = a_w * b_y + b_w * a_y + a_z * b_x - b_z * a_x
    q_res[..., 3] = a_w * b_z + b_w * a_z + a_x * b_y - b_x * a_y
    return q_res


@get_torch_jit_decorator()
def angular_distance_phi3(goal_quat, current_quat):
    """This function computes the angular distance phi_3.

    See Huynh, Du Q. "Metrics for 3D rotations: Comparison and analysis." Journal of Mathematical
    Imaging and Vision 35 (2009): 155-164.

    Args:
        goal_quat: _description_
        current_quat: _description_

    Returns:
        Angular distance in range [0,1]
    """
    dot_prod = (
        goal_quat[..., 0] * current_quat[..., 0]
        + goal_quat[..., 1] * current_quat[..., 1]
        + goal_quat[..., 2] * current_quat[..., 2]
        + goal_quat[..., 3] * current_quat[..., 3]
    )

    dot_prod = torch.abs(dot_prod)
    distance = dot_prod
    distance = torch.arccos(dot_prod) / (torch.pi * 0.5)
    return distance


class OrientationError(Function):
    @staticmethod
    def geodesic_distance(goal_quat, current_quat, quat_res):
        quat_grad, rot_error = geodesic_distance(goal_quat, current_quat, quat_res)
        return quat_grad, rot_error

    @staticmethod
    def forward(ctx, goal_quat, current_quat, quat_res):
        quat_grad, rot_error = OrientationError.geodesic_distance(goal_quat, current_quat, quat_res)
        ctx.save_for_backward(quat_grad)
        return rot_error

    @staticmethod
    def backward(ctx, grad_out):
        grad_mul = grad_mul1 = None
        (quat_grad,) = ctx.saved_tensors

        if ctx.needs_input_grad[1]:
            grad_mul = grad_out * quat_grad
        if ctx.needs_input_grad[0]:
            grad_mul1 = -1.0 * grad_out * quat_grad
        return grad_mul1, grad_mul, None


@get_torch_jit_decorator()
def normalize_quaternion(in_quaternion: torch.Tensor) -> torch.Tensor:
    k = torch.sign(in_quaternion[..., 0:1])
    # NOTE: torch sign returns 0 as sign value when value is 0.0
    k = torch.where(k == 0, 1.0, k)
    k2 = k / torch.linalg.norm(in_quaternion, dim=-1, keepdim=True)
    # normalize quaternion
    in_q = k2 * in_quaternion
    return in_q


@get_torch_jit_decorator()
def geodesic_distance(goal_quat, current_quat, quat_res):
    conjugate_quat = current_quat.detach().clone()
    conjugate_quat[..., 1:] *= -1.0
    quat_res = quat_multiply(goal_quat, conjugate_quat, quat_res)
    sign = torch.sign(quat_res[..., 0])
    sign = torch.where(sign == 0, 1.0, sign)
    quat_res = -1.0 * quat_res * sign.unsqueeze(-1)
    quat_res[..., 0] = 0.0
    rot_error = torch.norm(quat_res, dim=-1, keepdim=True)
    scale = 1.0 / rot_error
    scale = torch.nan_to_num(scale, 0.0, 0.0, 0.0)
    quat_res = quat_res * scale
    return quat_res, rot_error
