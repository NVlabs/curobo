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
"""
Implements differentiable point and pose transformations leveraging Warp kernels.
Most of these implementations are available through :class:`~curobo.types.math.Pose`.
"""
# Standard Library
from typing import Optional, Tuple

# Third Party
import torch
import warp as wp

# CuRobo
from curobo.util.logger import log_error
from curobo.util.torch_utils import get_torch_jit_decorator
from curobo.util.warp import init_warp


def transform_points(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    points: torch.Tensor,
    out_points: Optional[torch.Tensor] = None,
    out_gp: Optional[torch.Tensor] = None,
    out_gq: Optional[torch.Tensor] = None,
    out_gpt: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Transforms the given points using the provided position and quaternion.

    Args:
        position: The position tensor representing the translation of the transformation.
        quaternion: The quaternion tensor representing the rotation of the transformation.
            Quaternion format is [w, x, y, z].
        points: The points to be transformed.
        out_points: If provided, the transformed points will be stored in this tensor. If not
            provided, a new tensor will be created.
        out_gp: If provided, the gradient of the transformed points with respect to the position
            will be stored in this tensor. If not provided, a new tensor will be created.
        out_gq: If provided, the gradient of the transformed points with respect to the quaternion
            will be stored in this tensor. If not provided, a new tensor will be created.
        out_gpt: If provided, the gradient of the transformed points with respect to the original
            points will be stored in this tensor. If not provided, a new tensor will be created.

    Returns:
        torch.Tensor: The transformed points.
    """

    if out_points is None:
        out_points = torch.zeros((points.shape[0], 3), device=points.device, dtype=points.dtype)
    if out_gp is None:
        out_gp = torch.zeros((position.shape[0], 3), device=position.device, dtype=points.dtype)
    if out_gq is None:
        out_gq = torch.zeros((quaternion.shape[0], 4), device=quaternion.device, dtype=points.dtype)
    if out_gpt is None:
        out_gpt = torch.zeros((points.shape[0], 3), device=position.device, dtype=points.dtype)
    out_points = TransformPoint.apply(
        position, quaternion, points, out_points, out_gp, out_gq, out_gpt
    )
    return out_points


def batch_transform_points(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    points: torch.Tensor,
    out_points: Optional[torch.Tensor] = None,
    out_gp: Optional[torch.Tensor] = None,
    out_gq: Optional[torch.Tensor] = None,
    out_gpt: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Transforms the given points using the provided batch of position and quaternion.

    Args:
        position: The position tensor representing the translation of the transformation. Shape
            should be (batch_size, 3).
        quaternion: The quaternion tensor representing the rotation of the transformation.
            Quaternion format is [w, x, y, z]. Shape should be (batch_size, 4).
        points: The points to be transformed. Shape should be (batch_size, num_points, 3).
        out_points: If provided, the transformed points will be stored in this tensor. If not
            provided, a new tensor will be created.
        out_gp: If provided, the gradient of the transformed points with respect to the position
            will be stored in this tensor. If not provided, a new tensor will be created.
        out_gq: If provided, the gradient of the transformed points with respect to the quaternion
            will be stored in this tensor. If not provided, a new tensor will be created.
        out_gpt: If provided, the gradient of the transformed points with respect to the original
            points will be stored in this tensor. If not provided, a new tensor will be created.

    Returns:
        torch.Tensor: The transformed points with shape (batch_size, num_points, 3).
    """
    if out_points is None:
        out_points = torch.zeros(
            (points.shape[0], points.shape[1], 3), device=points.device, dtype=points.dtype
        )
    if out_gp is None:
        out_gp = torch.zeros((position.shape[0], 3), device=position.device, dtype=points.dtype)
    if out_gq is None:
        out_gq = torch.zeros((quaternion.shape[0], 4), device=quaternion.device, dtype=points.dtype)
    if out_gpt is None:
        out_gpt = torch.zeros(
            (points.shape[0], points.shape[1], 3), device=position.device, dtype=points.dtype
        )
    out_points = BatchTransformPoint.apply(
        position, quaternion, points, out_points, out_gp, out_gq, out_gpt
    )
    return out_points


@get_torch_jit_decorator()
def get_inv_transform(
    w_rot_c: torch.Tensor, w_trans_c: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the inverse of the given transformation.

    Args:
        w_rot_c: Rotation matrix in world frame.
        w_trans_c: Translation vector in world frame.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The inverse rotation matrix and translation vector.
    """
    c_rot_w = w_rot_c.transpose(-1, -2)
    c_trans_w = -1.0 * (c_rot_w @ w_trans_c.unsqueeze(-1)).squeeze(-1)
    return c_rot_w, c_trans_w


@get_torch_jit_decorator()
def transform_point_inverse(
    point: torch.Tensor, rot: torch.Tensor, trans: torch.Tensor
) -> torch.Tensor:
    """Transforms the given point using the inverse of the provided transformation.

    Args:
        point: Input point to be transformed.
        rot: Rotation matrix.
        trans: Translation vector.

    Returns:
        torch.Tensor: The transformed point.
    """
    # new_point = (rot @ (point).unsqueeze(-1)).squeeze(-1) + trans
    n_rot, n_trans = get_inv_transform(rot, trans)
    new_point = (point @ n_rot.transpose(-1, -2)) + n_trans
    return new_point


def matrix_to_quaternion(
    matrix: torch.Tensor,
    out_quat: Optional[torch.Tensor] = None,
    adj_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Converts the given rotation matrix to quaternion.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        out_quat: Output tensor to store the quaternions. If not provided, a new tensor will be
            created.
        adj_matrix: Gradient tensor, if not provided, a new tensor will be created.

    Returns:
        torch.Tensor: Quaternions with real part first, as tensor of shape (..., 4) [qw, qx,qy,qz].
    """
    matrix = matrix.view(-1, 3, 3)
    out_quat = MatrixToQuaternion.apply(matrix, out_quat, adj_matrix)
    return out_quat


def quaternion_to_matrix(
    quaternions: torch.Tensor,
    out_mat: Optional[torch.Tensor] = None,
    adj_quaternion: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert quaternion to rotation matrix.

    Args:
        quaternions: Input quaternions with real part first, as tensor of shape (..., 4).
        out_mat: Output rotation matrices as tensor of shape (..., 3, 3). If not provided, a new
            tensor will be created.
        adj_quaternion: Gradient tensor, if not provided, a new tensor will be created.

    Returns:
        torch.Tensor: Rotation matrices as tensor of shape (..., 3, 3).
    """
    # return torch_quaternion_to_matrix(quaternions)
    out_mat = QuatToMatrix.apply(quaternions, out_mat, adj_quaternion)
    return out_mat


def torch_quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first, as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    quaternions = torch.as_tensor(quaternions)
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def pose_to_matrix(
    position: torch.Tensor, quaternion: torch.Tensor, out_matrix: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Converts the given pose to a transformation matrix.

    Args:
        position: The position tensor representing the translation of the transformation.
        quaternion: The quaternion tensor representing the rotation of the transformation.
            Quaternion format is [w, x, y, z].
        out_matrix: If provided, the transformation matrix will be stored in this tensor. If not
            provided, a new tensor will be created.

    Returns:
        torch.Tensor: The transformation matrix.
    """

    if out_matrix is None:
        if len(position.shape) == 2:
            out_matrix = torch.zeros(
                (position.shape[0], 4, 4), device=position.device, dtype=position.dtype
            )
        else:
            out_matrix = torch.zeros(
                (position.shape[0], position.shape[1], 4, 4),
                device=position.device,
                dtype=position.dtype,
            )
        out_matrix[..., 3, 3] = 1.0
    out_matrix[..., :3, 3] = position
    out_matrix[..., :3, :3] = quaternion_to_matrix(quaternion)
    return out_matrix


def pose_multiply(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    position2: torch.Tensor,
    quaternion2: torch.Tensor,
    out_position: Optional[torch.Tensor] = None,
    out_quaternion: Optional[torch.Tensor] = None,
    adj_pos: Optional[torch.Tensor] = None,
    adj_quat: Optional[torch.Tensor] = None,
    adj_pos2: Optional[torch.Tensor] = None,
    adj_quat2: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multiplies two poses.

    The input poses can either be of shape (3,) or (batch_size, 3).

    Args:
        position: The position tensor representing the translation of the first transformation.
        quaternion: The quaternion tensor representing the rotation of the first transformation.
            The quaternion format is [w, x, y, z].
        position2: The position tensor representing the translation of the second transformation.
        quaternion2: The quaternion tensor representing the rotation of the second transformation.
        out_position: If provided, the position tensor of the multiplied pose will be stored in
            this tensor. If not provided, a new tensor will be created.
        out_quaternion: If provided, the quaternion tensor of the multiplied pose will be stored in
            this tensor. If not provided, a new tensor will be created.
        adj_pos: Gradient tensor for the position of the first pose. If not provided, a new tensor
            will be created.
        adj_quat: Gradient tensor for the quaternion of the first pose. If not provided, a new
            tensor will be created.
        adj_pos2: Gradient tensor for the position of the second pose. If not provided, a new
            tensor will be created.
        adj_quat2: Gradient tensor for the quaternion of the second pose. If not provided, a new
            tensor will be created.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The position and quaternion tensors of the multiplied
            pose.
    """
    if position.shape == position2.shape:
        out_position, out_quaternion = BatchTransformPose.apply(
            position,
            quaternion,
            position2,
            quaternion2,
            out_position,
            out_quaternion,
            adj_pos,
            adj_quat,
            adj_pos2,
            adj_quat2,
        )
    elif position.shape[0] == 1 and position2.shape[0] > 1:
        out_position, out_quaternion = TransformPose.apply(
            position,
            quaternion,
            position2,
            quaternion2,
            out_position,
            out_quaternion,
            adj_pos,
            adj_quat,
            adj_pos2,
            adj_quat2,
        )
    else:
        log_error("shapes not supported")

    return out_position, out_quaternion


def pose_inverse(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    out_position: Optional[torch.Tensor] = None,
    out_quaternion: Optional[torch.Tensor] = None,
    adj_pos: Optional[torch.Tensor] = None,
    adj_quat: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the inverse of the given pose.

    Args:
        position: The position tensor representing the translation of the transformation.
        quaternion: The quaternion tensor representing the rotation of the transformation.
        out_position: If provided, the position tensor of the inverse pose will be stored in this
            tensor. If not provided, a new tensor will be created.
        out_quaternion: If provided, the quaternion tensor of the inverse pose will be stored in
            this tensor. If not provided, a new tensor will be created.
        adj_pos: Gradient tensor for the position of the pose. If not provided, a new tensor will
            be created.
        adj_quat: Gradient tensor for the quaternion of the pose. If not provided, a new tensor
            will be created.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The position and quaternion tensors of the inverse pose.
    """

    out_position, out_quaternion = PoseInverse.apply(
        position,
        quaternion,
        out_position,
        out_quaternion,
        adj_pos,
        adj_quat,
    )

    return out_position, out_quaternion


@wp.kernel
def compute_pose_inverse(
    position: wp.array(dtype=wp.vec3),
    quat: wp.array(dtype=wp.vec4),
    out_position: wp.array(dtype=wp.vec3),
    out_quat: wp.array(dtype=wp.vec4),
):
    """Compute inverse of pose. This is a warp kernel.

    Args:
        position: Input position.
        quat: Input quaternion.
        out_position: Output position.
        out_quat: Output quaternion.
    """

    # b pose_1 and b pose_2, compute pose_1 * pose_2
    b_idx = wp.tid()
    # read data:

    in_position = position[b_idx]
    in_quat = quat[b_idx]

    # read point
    # create a transform from a vector/quaternion:
    t_1 = wp.transform(in_position, wp.quaternion(in_quat[1], in_quat[2], in_quat[3], in_quat[0]))
    t_3 = wp.transform_inverse(t_1)

    # write pt:
    out_q = wp.transform_get_rotation(t_3)

    out_v = wp.vec4(out_q[3], out_q[0], out_q[1], out_q[2])

    out_position[b_idx] = wp.transform_get_translation(t_3)
    out_quat[b_idx] = out_v


@wp.kernel
def compute_matrix_to_quat(
    in_mat: wp.array(dtype=wp.mat33),
    out_quat: wp.array(dtype=wp.vec4),
):
    """A warp kernel to convert rotation matrix to quaternion."""
    # b pose_1 and b pose_2, compute pose_1 * pose_2
    b_idx = wp.tid()
    # read data:

    in_m = in_mat[b_idx]

    # read point
    # create a transform from a vector/quaternion:
    out_q = wp.quat_from_matrix(in_m)

    out_v = wp.vec4(out_q[3], out_q[0], out_q[1], out_q[2])
    # write pt:
    out_quat[b_idx] = out_v


@wp.kernel
def compute_transform_point(
    position: wp.array(dtype=wp.vec3),
    quat: wp.array(dtype=wp.vec4),
    pt: wp.array(dtype=wp.vec3),
    n_pts: wp.int32,
    n_poses: wp.int32,
    out_pt: wp.array(dtype=wp.vec3),
):
    """A warp kernel to transform the given points using the provided position and quaternion."""
    # given n,3 points and b poses, get b,n,3 transformed points
    # we tile as
    tid = wp.tid()
    b_idx = tid / (n_pts)
    p_idx = tid - (b_idx * n_pts)

    # read data:

    in_position = position[b_idx]
    in_quat = quat[b_idx]
    in_pt = pt[p_idx]

    # read point
    # create a transform from a vector/quaternion:
    t = wp.transform(in_position, wp.quaternion(in_quat[1], in_quat[2], in_quat[3], in_quat[0]))

    # transform a point
    p = wp.transform_point(t, in_pt)

    # write pt:
    out_pt[b_idx * n_pts + p_idx] = p


@wp.kernel
def compute_batch_transform_point(
    position: wp.array(dtype=wp.vec3),
    quat: wp.array(dtype=wp.vec4),
    pt: wp.array(dtype=wp.vec3),
    n_pts: wp.int32,
    n_poses: wp.int32,
    out_pt: wp.array(dtype=wp.vec3),
):
    """A warp kernel to transform batch of points by batch of poses."""

    # given n,3 points and b poses, get b,n,3 transformed points
    # we tile as
    tid = wp.tid()
    b_idx = tid / (n_pts)
    p_idx = tid - (b_idx * n_pts)

    # read data:

    in_position = position[b_idx]
    in_quat = quat[b_idx]
    in_pt = pt[b_idx * n_pts + p_idx]

    # read point
    # create a transform from a vector/quaternion:
    t = wp.transform(in_position, wp.quaternion(in_quat[1], in_quat[2], in_quat[3], in_quat[0]))

    # transform a point
    p = wp.transform_point(t, in_pt)

    # write pt:
    out_pt[b_idx * n_pts + p_idx] = p


@wp.kernel
def compute_batch_transform_point_fp16(
    position: wp.array(dtype=wp.vec3h),
    quat: wp.array(dtype=wp.vec4h),
    pt: wp.array(dtype=wp.vec3h),
    n_pts: wp.int32,
    n_poses: wp.int32,
    out_pt: wp.array(dtype=wp.vec3h),
):
    """A warp kernel to transform batch of points by batch of poses."""

    # given n,3 points and b poses, get b,n,3 transformed points
    # we tile as
    tid = wp.tid()
    b_idx = tid / (n_pts)
    p_idx = tid - (b_idx * n_pts)

    # read data:

    in_position = position[b_idx]
    in_quat = quat[b_idx]
    in_pt = pt[b_idx * n_pts + p_idx]

    # read point
    # create a transform from a vector/quaternion:
    q_vec = wp.quaternion(in_quat[1], in_quat[2], in_quat[3], in_quat[0])
    t = wp.transformh(in_position, q_vec)

    # transform a point
    p = wp.transform_point(t, in_pt)

    # write pt:
    out_pt[b_idx * n_pts + p_idx] = p


@wp.kernel
def compute_batch_pose_multipy(
    position: wp.array(dtype=wp.vec3),
    quat: wp.array(dtype=wp.vec4),
    position2: wp.array(dtype=wp.vec3),
    quat2: wp.array(dtype=wp.vec4),
    out_position: wp.array(dtype=wp.vec3),
    out_quat: wp.array(dtype=wp.vec4),
):
    """A warp kernel multiplying two batch of poses."""
    # b pose_1 and b pose_2, compute pose_1 * pose_2
    b_idx = wp.tid()
    # read data:

    in_position = position[b_idx]
    in_quat = quat[b_idx]

    in_position2 = position2[b_idx]
    in_quat2 = quat2[b_idx]

    # read point
    # create a transform from a vector/quaternion:
    t_1 = wp.transform(in_position, wp.quaternion(in_quat[1], in_quat[2], in_quat[3], in_quat[0]))
    t_2 = wp.transform(
        in_position2, wp.quaternion(in_quat2[1], in_quat2[2], in_quat2[3], in_quat2[0])
    )

    # transform a point
    t_3 = wp.transform_multiply(t_1, t_2)

    # write pt:
    out_q = wp.transform_get_rotation(t_3)

    out_v = wp.vec4(out_q[3], out_q[0], out_q[1], out_q[2])

    out_position[b_idx] = wp.transform_get_translation(t_3)
    out_quat[b_idx] = out_v


@wp.kernel
def compute_quat_to_matrix(
    quat: wp.array(dtype=wp.vec4),
    out_mat: wp.array(dtype=wp.mat33),
):
    """A warp kernel to convert quaternion to rotation matrix."""
    # b pose_1 and b pose_2, compute pose_1 * pose_2
    b_idx = wp.tid()
    # read data:

    in_quat = quat[b_idx]

    # read point
    # create a transform from a vector/quaternion:
    q_1 = wp.quaternion(in_quat[1], in_quat[2], in_quat[3], in_quat[0])
    m_1 = wp.quat_to_matrix(q_1)

    # write pt:
    out_mat[b_idx] = m_1


@wp.kernel
def compute_pose_multipy(
    position: wp.array(dtype=wp.vec3),
    quat: wp.array(dtype=wp.vec4),
    position2: wp.array(dtype=wp.vec3),
    quat2: wp.array(dtype=wp.vec4),
    out_position: wp.array(dtype=wp.vec3),
    out_quat: wp.array(dtype=wp.vec4),
):
    """A warp kernel to multiply a batch of poses (position2) by a pose."""
    # b pose_1 and b pose_2, compute pose_1 * pose_2
    b_idx = wp.tid()
    # read data:

    in_position = position[0]
    in_quat = quat[0]

    in_position2 = position2[b_idx]
    in_quat2 = quat2[b_idx]

    # read point
    # create a transform from a vector/quaternion:
    t_1 = wp.transform(in_position, wp.quaternion(in_quat[1], in_quat[2], in_quat[3], in_quat[0]))
    t_2 = wp.transform(
        in_position2, wp.quaternion(in_quat2[1], in_quat2[2], in_quat2[3], in_quat2[0])
    )

    # transform a point
    t_3 = wp.transform_multiply(t_1, t_2)

    # write pt:
    out_q = wp.transform_get_rotation(t_3)

    out_v = wp.vec4(out_q[3], out_q[0], out_q[1], out_q[2])

    out_position[b_idx] = wp.transform_get_translation(t_3)
    out_quat[b_idx] = out_v


class TransformPoint(torch.autograd.Function):
    """A differentiable function to transform batch of points by a pose."""

    @staticmethod
    def forward(
        ctx,
        position: torch.Tensor,
        quaternion: torch.Tensor,
        points: torch.Tensor,
        out_points: torch.Tensor,
        adj_position: torch.Tensor,
        adj_quaternion: torch.Tensor,
        adj_points: torch.Tensor,
    ):
        n, _ = out_points.shape
        init_warp()
        ctx.save_for_backward(
            position, quaternion, points, out_points, adj_position, adj_quaternion, adj_points
        )
        b = 1
        ctx.b = b
        ctx.n = n

        wp.launch(
            kernel=compute_transform_point,
            dim=b * n,
            inputs=[
                wp.from_torch(position.detach().view(-1, 3).contiguous(), dtype=wp.vec3),
                wp.from_torch(quaternion.detach().view(-1, 4).contiguous(), dtype=wp.vec4),
                wp.from_torch(points.detach().view(-1, 3).contiguous(), dtype=wp.vec3),
                n,
                b,
            ],
            outputs=[wp.from_torch(out_points.view(-1, 3), dtype=wp.vec3)],
            stream=None if not position.is_cuda else wp.stream_from_torch(position.device),
            device=wp.device_from_torch(position.device),
        )

        return out_points

    @staticmethod
    def backward(ctx, grad_output):
        (
            position,
            quaternion,
            points,
            out_points,
            adj_position,
            adj_quaternion,
            adj_points,
        ) = ctx.saved_tensors
        adj_position = 0.0 * adj_position
        adj_quaternion = 0.0 * adj_quaternion
        adj_points = 0.0 * adj_points

        wp_adj_out_points = wp.from_torch(grad_output.view(-1, 3).contiguous(), dtype=wp.vec3)
        wp_adj_points = wp.from_torch(adj_points, dtype=wp.vec3)

        wp_adj_position = wp.from_torch(adj_position, dtype=wp.vec3)
        wp_adj_quat = wp.from_torch(adj_quaternion, dtype=wp.vec4)
        stream = None
        if position.is_cuda:
            stream = wp.stream_from_torch(position.device)
        wp.launch(
            kernel=compute_transform_point,
            dim=ctx.b * ctx.n,
            inputs=[
                wp.from_torch(
                    position.view(-1, 3).contiguous(), dtype=wp.vec3, grad=wp_adj_position
                ),
                wp.from_torch(quaternion.view(-1, 4).contiguous(), dtype=wp.vec4, grad=wp_adj_quat),
                wp.from_torch(points.view(-1, 3).contiguous(), dtype=wp.vec3, grad=wp_adj_points),
                ctx.n,
                ctx.b,
            ],
            outputs=[
                wp.from_torch(
                    out_points.view(-1, 3).contiguous(), dtype=wp.vec3, grad=wp_adj_out_points
                ),
            ],
            adj_inputs=[
                None,
                None,
                None,
                ctx.n,
                ctx.b,
            ],
            adj_outputs=[
                None,
            ],
            adjoint=True,
            stream=None if not grad_output.is_cuda else wp.stream_from_torch(grad_output.device),
            device=wp.device_from_torch(grad_output.device),
        )
        g_p = g_q = g_pt = None
        if ctx.needs_input_grad[0]:
            g_p = adj_position
        if ctx.needs_input_grad[1]:
            g_q = adj_quaternion
        if ctx.needs_input_grad[2]:
            g_pt = adj_points
        return g_p, g_q, g_pt, None, None, None, None


class BatchTransformPoint(torch.autograd.Function):
    """A differentiable function to transform batch of points by a batch of poses."""

    @staticmethod
    def forward(
        ctx,
        position: torch.Tensor,
        quaternion: torch.Tensor,
        points: torch.Tensor,
        out_points: torch.Tensor,
        adj_position: torch.Tensor,
        adj_quaternion: torch.Tensor,
        adj_points: torch.Tensor,
    ):
        b, n, _ = out_points.shape
        init_warp()
        points = points.detach()
        ctx.save_for_backward(
            position, quaternion, points, out_points, adj_position, adj_quaternion, adj_points
        )
        ctx.b = b
        ctx.n = n
        if points.dtype == torch.float32:
            wp.launch(
                kernel=compute_batch_transform_point,
                dim=b * n,
                inputs=[
                    wp.from_torch(
                        position.detach().view(-1, 3).contiguous(),
                        dtype=wp.types.vector(length=3, dtype=wp.float32),
                    ),
                    wp.from_torch(quaternion.detach().view(-1, 4).contiguous(), dtype=wp.vec4),
                    wp.from_torch(points.detach().view(-1, 3).contiguous(), dtype=wp.vec3),
                    n,
                    b,
                ],
                outputs=[wp.from_torch(out_points.view(-1, 3).contiguous(), dtype=wp.vec3)],
                stream=None if not position.is_cuda else wp.stream_from_torch(position.device),
                device=wp.device_from_torch(position.device),
            )
        elif points.dtype == torch.float16:
            wp.launch(
                kernel=compute_batch_transform_point_fp16,
                dim=b * n,
                inputs=[
                    wp.from_torch(
                        position.detach().view(-1, 3).contiguous(),
                        dtype=wp.types.vector(length=3, dtype=wp.float16),
                    ),
                    wp.from_torch(quaternion.detach().view(-1, 4).contiguous(), dtype=wp.vec4h),
                    wp.from_torch(points.detach().view(-1, 3).contiguous(), dtype=wp.vec3h),
                    n,
                    b,
                ],
                outputs=[wp.from_torch(out_points.view(-1, 3).contiguous(), dtype=wp.vec3h)],
                stream=None if not position.is_cuda else wp.stream_from_torch(position.device),
                device=wp.device_from_torch(position.device),
            )
        else:
            log_error("Unsupported dtype: " + str(points.dtype))

        return out_points

    @staticmethod
    def backward(ctx, grad_output):
        (
            position,
            quaternion,
            points,
            out_points,
            adj_position,
            adj_quaternion,
            adj_points,
        ) = ctx.saved_tensors
        init_warp()
        wp_adj_out_points = wp.from_torch(grad_output.view(-1, 3).contiguous(), dtype=wp.vec3)

        adj_position = 0.0 * adj_position
        adj_quaternion = 0.0 * adj_quaternion
        adj_points = 0.0 * adj_points

        wp_adj_points = wp.from_torch(adj_points.view(-1, 3), dtype=wp.vec3)
        wp_adj_position = wp.from_torch(adj_position.view(-1, 3), dtype=wp.vec3)
        wp_adj_quat = wp.from_torch(adj_quaternion.view(-1, 4), dtype=wp.vec4)
        wp.launch(
            kernel=compute_batch_transform_point,
            dim=ctx.b * ctx.n,
            inputs=[
                wp.from_torch(
                    position.view(-1, 3).contiguous(), dtype=wp.vec3, grad=wp_adj_position
                ),
                wp.from_torch(quaternion.view(-1, 4).contiguous(), dtype=wp.vec4, grad=wp_adj_quat),
                wp.from_torch(points.view(-1, 3).contiguous(), dtype=wp.vec3, grad=wp_adj_points),
                ctx.n,
                ctx.b,
            ],
            outputs=[
                wp.from_torch(out_points.view(-1, 3), dtype=wp.vec3, grad=wp_adj_out_points),
            ],
            adj_inputs=[
                None,
                None,
                None,
                ctx.n,
                ctx.b,
            ],
            adj_outputs=[
                None,
            ],
            adjoint=True,
            stream=None if not grad_output.is_cuda else wp.stream_from_torch(grad_output.device),
            device=wp.device_from_torch(grad_output.device),
        )
        g_p = g_q = g_pt = None
        if ctx.needs_input_grad[0]:
            g_p = adj_position
        if ctx.needs_input_grad[1]:
            g_q = adj_quaternion
        if ctx.needs_input_grad[2]:
            g_pt = adj_points
        return g_p, g_q, g_pt, None, None, None, None


class BatchTransformPose(torch.autograd.Function):
    """A differentiable function to transform batch of poses by a pose."""

    @staticmethod
    def forward(
        ctx,
        position: torch.Tensor,
        quaternion: torch.Tensor,
        position2: torch.Tensor,
        quaternion2: torch.Tensor,
        out_position: torch.Tensor,
        out_quaternion: torch.Tensor,
        adj_position: torch.Tensor,
        adj_quaternion: torch.Tensor,
        adj_position2: torch.Tensor,
        adj_quaternion2: torch.Tensor,
    ):
        b, _ = position.view(-1, 3).shape

        if out_position is None:
            out_position = torch.zeros_like(position2)
        if out_quaternion is None:
            out_quaternion = torch.zeros_like(quaternion2)
        if adj_position is None:
            adj_position = torch.zeros_like(position)
        if adj_quaternion is None:
            adj_quaternion = torch.zeros_like(quaternion)
        if adj_position2 is None:
            adj_position2 = torch.zeros_like(position2)
        if adj_quaternion2 is None:
            adj_quaternion2 = torch.zeros_like(quaternion2)

        init_warp()
        ctx.save_for_backward(
            position,
            quaternion,
            position2,
            quaternion2,
            out_position,
            out_quaternion,
            adj_position,
            adj_quaternion,
            adj_position2,
            adj_quaternion2,
        )
        ctx.b = b
        wp.launch(
            kernel=compute_batch_pose_multipy,
            dim=b,
            inputs=[
                wp.from_torch(position.detach().view(-1, 3).contiguous(), dtype=wp.vec3),
                wp.from_torch(quaternion.detach().view(-1, 4).contiguous(), dtype=wp.vec4),
                wp.from_torch(position2.detach().view(-1, 3).contiguous(), dtype=wp.vec3),
                wp.from_torch(quaternion2.detach().view(-1, 4).contiguous(), dtype=wp.vec4),
            ],
            outputs=[
                wp.from_torch(out_position.detach().view(-1, 3).contiguous(), dtype=wp.vec3),
                wp.from_torch(out_quaternion.detach().view(-1, 4).contiguous(), dtype=wp.vec4),
            ],
            stream=None if not position.is_cuda else wp.stream_from_torch(position.device),
            device=wp.device_from_torch(position.device),
        )

        return out_position, out_quaternion

    @staticmethod
    def backward(ctx, grad_out_position, grad_out_quaternion):
        (
            position,
            quaternion,
            position2,
            quaternion2,
            out_position,
            out_quaternion,
            adj_position,
            adj_quaternion,
            adj_position2,
            adj_quaternion2,
        ) = ctx.saved_tensors
        init_warp()

        wp_adj_out_position = wp.from_torch(
            grad_out_position.view(-1, 3).contiguous(), dtype=wp.vec3
        )
        wp_adj_out_quaternion = wp.from_torch(
            grad_out_quaternion.view(-1, 4).contiguous(), dtype=wp.vec4
        )

        adj_position = 0.0 * adj_position
        adj_quaternion = 0.0 * adj_quaternion
        adj_position2 = 0.0 * adj_position2
        adj_quaternion2 = 0.0 * adj_quaternion2

        wp_adj_position = wp.from_torch(adj_position.view(-1, 3), dtype=wp.vec3)
        wp_adj_quat = wp.from_torch(adj_quaternion.view(-1, 4), dtype=wp.vec4)
        wp_adj_position2 = wp.from_torch(adj_position2.view(-1, 3), dtype=wp.vec3)
        wp_adj_quat2 = wp.from_torch(adj_quaternion2.view(-1, 4), dtype=wp.vec4)

        wp.launch(
            kernel=compute_batch_pose_multipy,
            dim=ctx.b,
            inputs=[
                wp.from_torch(
                    position.view(-1, 3).contiguous(), dtype=wp.vec3, grad=wp_adj_position
                ),
                wp.from_torch(quaternion.view(-1, 4).contiguous(), dtype=wp.vec4, grad=wp_adj_quat),
                wp.from_torch(
                    position2.view(-1, 3).contiguous(), dtype=wp.vec3, grad=wp_adj_position2
                ),
                wp.from_torch(
                    quaternion2.view(-1, 4).contiguous(), dtype=wp.vec4, grad=wp_adj_quat2
                ),
            ],
            outputs=[
                wp.from_torch(
                    out_position.view(-1, 3).contiguous(), dtype=wp.vec3, grad=wp_adj_out_position
                ),
                wp.from_torch(
                    out_quaternion.view(-1, 4).contiguous(),
                    dtype=wp.vec4,
                    grad=wp_adj_out_quaternion,
                ),
            ],
            adj_inputs=[
                None,
                None,
                None,
                None,
            ],
            adj_outputs=[
                None,
                None,
            ],
            adjoint=True,
            stream=(
                None
                if not grad_out_position.is_cuda
                else wp.stream_from_torch(grad_out_position.device)
            ),
            device=wp.device_from_torch(grad_out_position.device),
        )
        g_p1 = g_q1 = g_p2 = g_q2 = None
        if ctx.needs_input_grad[0]:
            g_p1 = adj_position
        if ctx.needs_input_grad[1]:
            g_q1 = adj_quaternion
        if ctx.needs_input_grad[2]:
            g_p2 = adj_position2
        if ctx.needs_input_grad[3]:
            g_q2 = adj_quaternion2
        return g_p1, g_q1, g_p2, g_q2, None, None, None, None, None, None


class TransformPose(torch.autograd.Function):
    """A differentiable function to transform a batch of poses by another batch of poses."""

    @staticmethod
    def forward(
        ctx,
        position: torch.Tensor,
        quaternion: torch.Tensor,
        position2: torch.Tensor,
        quaternion2: torch.Tensor,
        out_position: torch.Tensor,
        out_quaternion: torch.Tensor,
        adj_position: torch.Tensor,
        adj_quaternion: torch.Tensor,
        adj_position2: torch.Tensor,
        adj_quaternion2: torch.Tensor,
    ):
        b, _ = position2.view(-1, 3).shape
        init_warp()
        if out_position is None:
            out_position = torch.zeros_like(position2)
        if out_quaternion is None:
            out_quaternion = torch.zeros_like(quaternion2)
        if adj_position is None:
            adj_position = torch.zeros_like(position)
        if adj_quaternion is None:
            adj_quaternion = torch.zeros_like(quaternion)
        if adj_position2 is None:
            adj_position2 = torch.zeros_like(position2)
        if adj_quaternion2 is None:
            adj_quaternion2 = torch.zeros_like(quaternion2)

        ctx.save_for_backward(
            position,
            quaternion,
            position2,
            quaternion2,
            out_position,
            out_quaternion,
            adj_position,
            adj_quaternion,
            adj_position2,
            adj_quaternion2,
        )
        ctx.b = b
        wp.launch(
            kernel=compute_pose_multipy,
            dim=b,
            inputs=[
                wp.from_torch(position.detach().view(-1, 3).contiguous(), dtype=wp.vec3),
                wp.from_torch(quaternion.detach().view(-1, 4).contiguous(), dtype=wp.vec4),
                wp.from_torch(position2.detach().view(-1, 3).contiguous(), dtype=wp.vec3),
                wp.from_torch(quaternion2.detach().view(-1, 4).contiguous(), dtype=wp.vec4),
            ],
            outputs=[
                wp.from_torch(out_position.detach().view(-1, 3).contiguous(), dtype=wp.vec3),
                wp.from_torch(out_quaternion.detach().view(-1, 4).contiguous(), dtype=wp.vec4),
            ],
            stream=(None if not position.is_cuda else wp.stream_from_torch(position.device)),
            device=wp.device_from_torch(position.device),
        )

        return out_position, out_quaternion

    @staticmethod
    def backward(ctx, grad_out_position, grad_out_quaternion):
        (
            position,
            quaternion,
            position2,
            quaternion2,
            out_position,
            out_quaternion,
            adj_position,
            adj_quaternion,
            adj_position2,
            adj_quaternion2,
        ) = ctx.saved_tensors
        init_warp()

        wp_adj_out_position = wp.from_torch(
            grad_out_position.view(-1, 3).contiguous(), dtype=wp.vec3
        )
        wp_adj_out_quaternion = wp.from_torch(
            grad_out_quaternion.view(-1, 4).contiguous(), dtype=wp.vec4
        )

        adj_position = 0.0 * adj_position
        adj_quaternion = 0.0 * adj_quaternion
        adj_position2 = 0.0 * adj_position2
        adj_quaternion2 = 0.0 * adj_quaternion2

        wp_adj_position = wp.from_torch(adj_position.view(-1, 3), dtype=wp.vec3)
        wp_adj_quat = wp.from_torch(adj_quaternion.view(-1, 4), dtype=wp.vec4)
        wp_adj_position2 = wp.from_torch(adj_position2.view(-1, 3), dtype=wp.vec3)
        wp_adj_quat2 = wp.from_torch(adj_quaternion2.view(-1, 4), dtype=wp.vec4)

        wp.launch(
            kernel=compute_batch_pose_multipy,
            dim=ctx.b,
            inputs=[
                wp.from_torch(
                    position.view(-1, 3).contiguous(), dtype=wp.vec3, grad=wp_adj_position
                ),
                wp.from_torch(quaternion.view(-1, 4).contiguous(), dtype=wp.vec4, grad=wp_adj_quat),
                wp.from_torch(
                    position2.view(-1, 3).contiguous(), dtype=wp.vec3, grad=wp_adj_position2
                ),
                wp.from_torch(
                    quaternion2.view(-1, 4).contiguous(), dtype=wp.vec4, grad=wp_adj_quat2
                ),
            ],
            outputs=[
                wp.from_torch(
                    out_position.view(-1, 3).contiguous(), dtype=wp.vec3, grad=wp_adj_out_position
                ),
                wp.from_torch(
                    out_quaternion.view(-1, 4).contiguous(),
                    dtype=wp.vec4,
                    grad=wp_adj_out_quaternion,
                ),
            ],
            adj_inputs=[
                None,
                None,
                None,
                None,
            ],
            adj_outputs=[
                None,
                None,
            ],
            stream=(
                None
                if not grad_out_position.is_cuda
                else wp.stream_from_torch(grad_out_position.device)
            ),
            device=wp.device_from_torch(grad_out_position.device),
            adjoint=True,
        )
        g_p1 = g_q1 = g_p2 = g_q2 = None
        if ctx.needs_input_grad[0]:
            g_p1 = adj_position
        if ctx.needs_input_grad[1]:
            g_q1 = adj_quaternion
        if ctx.needs_input_grad[2]:
            g_p2 = adj_position2
        if ctx.needs_input_grad[3]:
            g_q2 = adj_quaternion2
        return g_p1, g_q1, g_p2, g_q2, None, None, None, None, None, None


class PoseInverse(torch.autograd.Function):
    """A differentiable function to get the inverse of a pose (also supports batch)."""

    @staticmethod
    def forward(
        ctx,
        position: torch.Tensor,
        quaternion: torch.Tensor,
        out_position: torch.Tensor,
        out_quaternion: torch.Tensor,
        adj_position: torch.Tensor,
        adj_quaternion: torch.Tensor,
    ):

        if out_position is None:
            out_position = torch.zeros_like(position)
        if out_quaternion is None:
            out_quaternion = torch.zeros_like(quaternion)
        if adj_position is None:
            adj_position = torch.zeros_like(position)
        if adj_quaternion is None:
            adj_quaternion = torch.zeros_like(quaternion)
        b, _ = position.view(-1, 3).shape
        ctx.b = b
        init_warp()
        ctx.save_for_backward(
            position,
            quaternion,
            out_position,
            out_quaternion,
            adj_position,
            adj_quaternion,
        )

        wp.launch(
            kernel=compute_pose_inverse,
            dim=b,
            inputs=[
                wp.from_torch(position.detach().view(-1, 3).contiguous(), dtype=wp.vec3),
                wp.from_torch(quaternion.detach().view(-1, 4).contiguous(), dtype=wp.vec4),
            ],
            outputs=[
                wp.from_torch(out_position.detach().view(-1, 3).contiguous(), dtype=wp.vec3),
                wp.from_torch(out_quaternion.detach().view(-1, 4).contiguous(), dtype=wp.vec4),
            ],
            stream=(None if not position.is_cuda else wp.stream_from_torch(position.device)),
            device=wp.device_from_torch(position.device),
        )

        return out_position, out_quaternion

    @staticmethod
    def backward(ctx, grad_out_position, grad_out_quaternion):
        (
            position,
            quaternion,
            out_position,
            out_quaternion,
            adj_position,
            adj_quaternion,
        ) = ctx.saved_tensors
        init_warp()

        wp_adj_out_position = wp.from_torch(
            grad_out_position.view(-1, 3).contiguous(), dtype=wp.vec3
        )
        wp_adj_out_quaternion = wp.from_torch(
            grad_out_quaternion.view(-1, 4).contiguous(), dtype=wp.vec4
        )

        adj_position = 0.0 * adj_position
        adj_quaternion = 0.0 * adj_quaternion

        wp_adj_position = wp.from_torch(adj_position.view(-1, 3), dtype=wp.vec3)
        wp_adj_quat = wp.from_torch(adj_quaternion.view(-1, 4), dtype=wp.vec4)

        wp.launch(
            kernel=compute_pose_inverse,
            dim=ctx.b,
            inputs=[
                wp.from_torch(
                    position.view(-1, 3).contiguous(), dtype=wp.vec3, grad=wp_adj_position
                ),
                wp.from_torch(quaternion.view(-1, 4).contiguous(), dtype=wp.vec4, grad=wp_adj_quat),
            ],
            outputs=[
                wp.from_torch(
                    out_position.view(-1, 3).contiguous(), dtype=wp.vec3, grad=wp_adj_out_position
                ),
                wp.from_torch(
                    out_quaternion.view(-1, 4).contiguous(),
                    dtype=wp.vec4,
                    grad=wp_adj_out_quaternion,
                ),
            ],
            adj_inputs=[
                None,
                None,
            ],
            adj_outputs=[
                None,
                None,
            ],
            adjoint=True,
            stream=(
                None if not out_position.is_cuda else wp.stream_from_torch(out_position.device)
            ),
            device=wp.device_from_torch(out_position.device),
        )
        g_p1 = g_q1 = None
        if ctx.needs_input_grad[0]:
            g_p1 = adj_position
        if ctx.needs_input_grad[1]:
            g_q1 = adj_quaternion

        return g_p1, g_q1, None, None, None, None


class QuatToMatrix(torch.autograd.Function):
    """A differentiable function for converting quaternions to rotation matrices."""

    @staticmethod
    def forward(
        ctx,
        quaternion: torch.Tensor,
        out_mat: torch.Tensor,
        adj_quaternion: torch.Tensor,
    ):
        b, _ = quaternion.shape

        if out_mat is None:
            out_mat = torch.zeros(
                (quaternion.shape[0], 3, 3), device=quaternion.device, dtype=quaternion.dtype
            )
        if adj_quaternion is None:
            adj_quaternion = torch.zeros_like(quaternion)

        init_warp()
        ctx.save_for_backward(
            quaternion,
            out_mat,
            adj_quaternion,
        )
        ctx.b = b

        wp.launch(
            kernel=compute_quat_to_matrix,
            dim=b,
            inputs=[
                wp.from_torch(quaternion.detach().view(-1, 4).contiguous(), dtype=wp.vec4),
            ],
            outputs=[
                wp.from_torch(out_mat.detach().view(-1, 3, 3).contiguous(), dtype=wp.mat33),
            ],
            stream=(None if not quaternion.is_cuda else wp.stream_from_torch(quaternion.device)),
            device=wp.device_from_torch(quaternion.device),
        )

        return out_mat

    @staticmethod
    def backward(ctx, grad_out_mat):
        (
            quaternion,
            out_mat,
            adj_quaternion,
        ) = ctx.saved_tensors
        init_warp()

        wp_adj_out_mat = wp.from_torch(grad_out_mat.view(-1, 3, 3).contiguous(), dtype=wp.mat33)

        adj_quaternion[:] = 0.0 * adj_quaternion

        wp_adj_quat = wp.from_torch(adj_quaternion.view(-1, 4), dtype=wp.vec4)

        wp.launch(
            kernel=compute_quat_to_matrix,
            dim=ctx.b,
            inputs=[
                wp.from_torch(quaternion.view(-1, 4).contiguous(), dtype=wp.vec4, grad=wp_adj_quat),
            ],
            outputs=[
                wp.from_torch(
                    out_mat.view(-1, 3, 3).contiguous(),
                    dtype=wp.mat33,
                    grad=wp_adj_out_mat,
                ),
            ],
            adj_inputs=[
                None,
            ],
            adj_outputs=[
                None,
            ],
            adjoint=True,
            stream=(
                None if not grad_out_mat.is_cuda else wp.stream_from_torch(grad_out_mat.device)
            ),
            device=wp.device_from_torch(grad_out_mat.device),
        )
        g_q1 = None
        if ctx.needs_input_grad[0]:
            g_q1 = adj_quaternion

        return g_q1, None, None


class MatrixToQuaternion(torch.autograd.Function):
    """A differentiable function for converting rotation matrices to quaternions."""

    @staticmethod
    def forward(
        ctx,
        in_mat: torch.Tensor,
        out_quaternion: torch.Tensor,
        adj_mat: torch.Tensor,
    ):
        b, _, _ = in_mat.shape

        if out_quaternion is None:
            out_quaternion = torch.zeros(
                (in_mat.shape[0], 4), device=in_mat.device, dtype=in_mat.dtype
            )
        if adj_mat is None:
            adj_mat = torch.zeros_like(in_mat)

        init_warp()
        ctx.save_for_backward(
            in_mat,
            out_quaternion,
            adj_mat,
        )
        ctx.b = b

        wp.launch(
            kernel=compute_matrix_to_quat,
            dim=b,
            inputs=[
                wp.from_torch(in_mat.detach().view(-1, 3, 3).contiguous(), dtype=wp.mat33),
            ],
            outputs=[
                wp.from_torch(out_quaternion.detach().view(-1, 4).contiguous(), dtype=wp.vec4),
            ],
            stream=(None if not in_mat.is_cuda else wp.stream_from_torch(in_mat.device)),
            device=wp.device_from_torch(in_mat.device),
        )

        return out_quaternion

    @staticmethod
    def backward(ctx, grad_out_q):
        (
            in_mat,
            out_quaternion,
            adj_mat,
        ) = ctx.saved_tensors
        init_warp()

        wp_adj_out_q = wp.from_torch(grad_out_q.view(-1, 4).contiguous(), dtype=wp.vec4)

        adj_mat = 0.0 * adj_mat

        wp_adj_mat = wp.from_torch(adj_mat.view(-1, 3, 3), dtype=wp.mat33)

        wp.launch(
            kernel=compute_matrix_to_quat,
            dim=ctx.b,
            inputs=[
                wp.from_torch(in_mat.view(-1, 3, 3).contiguous(), dtype=wp.mat33, grad=wp_adj_mat),
            ],
            outputs=[
                wp.from_torch(
                    out_quaternion.view(-1, 4).contiguous(), dtype=wp.vec4, grad=wp_adj_out_q
                ),
            ],
            adj_inputs=[
                None,
            ],
            adj_outputs=[
                None,
            ],
            adjoint=True,
            stream=(None if not in_mat.is_cuda else wp.stream_from_torch(in_mat.device)),
            device=wp.device_from_torch(in_mat.device),
        )
        g_q1 = None
        if ctx.needs_input_grad[0]:
            g_q1 = adj_mat

        return g_q1, None, None
