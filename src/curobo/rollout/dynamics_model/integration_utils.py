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
# Standard Library
from typing import List

# Third Party
import torch
from packaging import version

# CuRobo
from curobo.curobolib.tensor_step import (
    tensor_step_acc_fwd,
    tensor_step_acc_idx_fwd,
    tensor_step_pos_clique_bwd,
    tensor_step_pos_clique_fwd,
    tensor_step_pos_clique_idx_fwd,
)
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState
from curobo.util.torch_utils import get_torch_jit_decorator


def build_clique_matrix(horizon, dt, device="cpu", dtype=torch.float32):
    diag_dt = torch.diag(1 / dt)
    one_t = torch.ones(horizon - 1, device=device, dtype=dtype)
    fd_mat_pos = torch.diag_embed(one_t, offset=-1)

    fd_mat_vel = -1.0 * torch.diag_embed(one_t, offset=-1)
    one_t = torch.ones(horizon - 1, device=device, dtype=dtype)

    fd_mat_vel += torch.eye(horizon, device=device, dtype=dtype)
    fd_mat_vel[0, 0] = 0.0
    fd_mat_vel = diag_dt @ fd_mat_vel
    fd_mat_acc = diag_dt @ fd_mat_vel.clone()

    fd_mat = torch.cat((fd_mat_pos, fd_mat_vel, fd_mat_acc), dim=0)
    return fd_mat


def build_fd_matrix(
    horizon,
    device="cpu",
    dtype=torch.float32,
    order=1,
    PREV_STATE=False,
    FULL_RANK=False,
    SHIFT=False,
):
    if PREV_STATE:
        # build order 1 fd matrix of horizon+order size
        fd1_mat = build_fd_matrix(horizon + order, device, dtype, order=1)

        # multiply order times to get fd_order matrix [h+order, h+order]
        fd_mat = fd1_mat
        fd_single = fd_mat.clone()
        for _ in range(order - 1):
            fd_mat = fd_single @ fd_mat
        # return [horizon,h+order]
        fd_mat = -1.0 * fd_mat[:horizon, :]
        # fd_mat = torch.zeros((horizon, horizon + order),device=device, dtype=dtype)
        # one_t = torch.ones(horizon, device=device, dtype=dtype)
        # fd_mat[:horizon, :horizon] = torch.diag_embed(one_t)
        # print(torch.diag_embed(one_t, offset=1).shape, fd_mat.shape)
        # fd_mat += - torch.diag_embed(one_t, offset=1)[:-1,:]

    elif FULL_RANK:
        fd_mat = torch.eye(horizon, device=device, dtype=dtype)

        one_t = torch.ones(horizon // 2, device=device, dtype=dtype)
        fd_mat[: horizon // 2, : horizon // 2] = torch.diag_embed(one_t)
        fd_mat[: horizon // 2 + 1, : horizon // 2 + 1] += -torch.diag_embed(one_t, offset=1)
        one_t = torch.ones(horizon // 2, device=device, dtype=dtype)
        fd_mat[horizon // 2 :, horizon // 2 :] += -torch.diag_embed(one_t, offset=-1)
        fd_mat[horizon // 2, horizon // 2] = 0.0
        fd_mat[horizon // 2, horizon // 2 - 1] = -1.0
        fd_mat[horizon // 2, horizon // 2 + 1] = 1.0
    else:
        fd_mat = torch.zeros((horizon, horizon), device=device, dtype=dtype)
        if horizon > 1:
            one_t = torch.ones(horizon - 1, device=device, dtype=dtype)
            if not SHIFT:
                fd_mat[: horizon - 1, : horizon - 1] = -1.0 * torch.diag_embed(one_t)
                fd_mat += torch.diag_embed(one_t, offset=1)
            else:
                fd_mat[1:, : horizon - 1] = -1.0 * torch.diag_embed(one_t)
                fd_mat[1:, 1:] += torch.diag_embed(one_t)
            fd_og = fd_mat.clone()
            for _ in range(order - 1):
                fd_mat = fd_og @ fd_mat
            # if order > 1:
            #    #print(order, fd_mat)
            #    for i in range(order):
            #        fd_mat[i,:] /= (2**(i+2))
            #        #print(order, fd_mat[order])
            #    #print(order, fd_mat)

            # fd_mat[:order]
            # if order > 1:
            #    fd_mat[:order-1, :] = 0.0

            # recreate this as a sparse tensor?
            # print(fd_mat)
            # sparse_indices = []
            # sparse_values = []
            # for i in range(horizon-1):
            #    sparse_indices.extend([[i,i], [i,i+1]])
            #    sparse_values.extend([-1.0, 1.0])
            # sparse_indices.extend([[horizon-1, horizon-1]])
            # sparse_values.extend([0.0])
            # fd_kernel = torch.sparse_coo_tensor(torch.tensor(sparse_indices).t(),
            # torch.tensor(sparse_values), device=device, dtype=dtype)
            # fd_mat = fd_kernel.to_dense()
    return fd_mat


def build_int_matrix(horizon, diagonal=0, device="cpu", dtype=torch.float32, order=1, traj_dt=None):
    integrate_matrix = torch.tril(
        torch.ones((horizon, horizon), device=device, dtype=dtype), diagonal=diagonal
    )
    chain_list = [torch.eye(horizon, device=device, dtype=dtype)]
    if traj_dt is None:
        chain_list.extend([integrate_matrix for i in range(order)])
    else:
        diag_dt = torch.diag(traj_dt)

        for _ in range(order):
            chain_list.append(integrate_matrix)
            chain_list.append(diag_dt)
    if len(chain_list) == 1:
        integrate_matrix = chain_list[0]
    elif version.parse(torch.__version__) < version.parse("1.9.0"):
        integrate_matrix = torch.chain_matmul(*chain_list)
    else:
        integrate_matrix = torch.linalg.multi_dot(chain_list)

    return integrate_matrix


def build_start_state_mask(horizon, tensor_args: TensorDeviceType):
    mask = torch.zeros((horizon, 1), device=tensor_args.device, dtype=tensor_args.dtype)
    # n_mask = torch.eye(horizon, device=tensor_args.device, dtype=tensor_args.dtype)
    n_mask = torch.diag_embed(
        torch.ones((horizon - 1), device=tensor_args.device, dtype=tensor_args.dtype), offset=-1
    )
    mask[0, 0] = 1.0
    # n_mask[0,0] = 0.0
    return mask, n_mask


# @get_torch_jit_decorator()
def tensor_step_jerk(state, act, state_seq, dt_h, n_dofs, integrate_matrix, fd_matrix=None):
    # type: (Tensor, Tensor, Tensor, Tensor, int, Tensor, Optional[Tensor]) -> Tensor

    # This is batch,n_dof
    q = state[:, :n_dofs]
    qd = state[:, n_dofs : 2 * n_dofs]
    qdd = state[:, 2 * n_dofs : 3 * n_dofs]

    diag_dt = torch.diag(dt_h)
    # qd_new = act
    # integrate velocities:
    qdd_new = qdd + torch.matmul(integrate_matrix, torch.matmul(diag_dt, act))
    qd_new = qd + torch.matmul(integrate_matrix, torch.matmul(diag_dt, qdd_new))
    q_new = q + torch.matmul(integrate_matrix, torch.matmul(diag_dt, qd_new))
    state_seq[:, :, :n_dofs] = q_new
    state_seq[:, :, n_dofs : n_dofs * 2] = qd_new
    state_seq[:, :, n_dofs * 2 : n_dofs * 3] = qdd_new

    return state_seq


# @get_torch_jit_decorator()
def euler_integrate(q_0, u, diag_dt, integrate_matrix):
    # q_new = q_0 + torch.matmul(integrate_matrix, torch.matmul(diag_dt, u))
    q_new = q_0 + torch.matmul(integrate_matrix, u)
    # q_new = torch.addmm(q_0,integrate_matrix,torch.matmul(diag_dt, u))
    return q_new


# @get_torch_jit_decorator()
def tensor_step_acc(state, act, state_seq, dt_h, n_dofs, integrate_matrix, fd_matrix=None):
    # type: (Tensor, Tensor, Tensor, Tensor, int, Tensor, Optional[Tensor]) -> Tensor
    # This is batch,n_dof
    q = state[..., :n_dofs]
    qd = state[..., n_dofs : 2 * n_dofs]
    qdd_new = act
    diag_dt = torch.diag(dt_h)
    diag_dt_2 = torch.diag(dt_h**2)
    qd_new = euler_integrate(qd, qdd_new, diag_dt, integrate_matrix)
    q_new = euler_integrate(q, qd_new, diag_dt, integrate_matrix)

    state_seq[..., n_dofs * 2 : n_dofs * 3] = qdd_new
    state_seq[..., n_dofs : n_dofs * 2] = qd_new
    # state_seq[:,1:, n_dofs: n_dofs * 2] = qd_new[:,:-1,:]
    # state_seq[:,0:1, n_dofs: n_dofs * 2] = qd
    # state_seq[:,1:, :n_dofs] = q_new[:,:-1,:] #+ 0.5 * torch.matmul(diag_dt_2,qdd_new)
    state_seq[..., :n_dofs] = q_new  # + 0.5 * torch.matmul(diag_dt_2,qdd_new)
    # state_seq[:,0:1, :n_dofs] = q #state[...,:n_dofs]

    return state_seq


@get_torch_jit_decorator()
def jit_tensor_step_pos_clique_contiguous(pos_act, start_position, mask, n_mask, fd_1, fd_2, fd_3):
    state_position = (start_position.unsqueeze(1).transpose(1, 2) @ mask.transpose(0, 1)) + (
        pos_act.transpose(1, 2) @ n_mask.transpose(0, 1)
    )
    # state_position = mask @ start_position.unsqueeze(1) + n_mask @ pos_act
    # print(state_position.shape, fd_1.shape)
    # # below 3 can be done in parallel:
    state_vel = (state_position @ fd_1.transpose(0, 1)).transpose(1, 2).contiguous()
    state_acc = (state_position @ fd_2.transpose(0, 1)).transpose(1, 2).contiguous()
    state_jerk = (state_position @ fd_3.transpose(0, 1)).transpose(1, 2).contiguous()
    state_position = state_position.transpose(1, 2).contiguous()
    return state_position, state_vel, state_acc, state_jerk


@get_torch_jit_decorator()
def jit_tensor_step_pos_clique(pos_act, start_position, mask, n_mask, fd_1, fd_2, fd_3):
    state_position = mask @ start_position.unsqueeze(1) + n_mask @ pos_act
    state_vel = fd_1 @ state_position
    state_acc = fd_2 @ state_position
    state_jerk = fd_3 @ state_position
    return state_position, state_vel, state_acc, state_jerk


@get_torch_jit_decorator()
def jit_backward_pos_clique(grad_p, grad_v, grad_a, grad_j, n_mask, fd_1, fd_2, fd_3):
    p_grad = (
        grad_p
        + (fd_3).transpose(-1, -2) @ grad_j
        + (fd_2).transpose(-1, -2) @ grad_a
        + (fd_1).transpose(-1, -2) @ grad_v
    )
    u_grad = (n_mask).transpose(-1, -2) @ p_grad
    # u_grad = n_mask @ p_grad
    # p_grad = fd_3 @ grad_j + fd_2 @ grad_a + fd_1 @ grad_v + grad_p
    # u_grad = n_mask @ p_grad

    return u_grad


@get_torch_jit_decorator()
def jit_backward_pos_clique_contiguous(grad_p, grad_v, grad_a, grad_j, n_mask, fd_1, fd_2, fd_3):
    p_grad = grad_p + (
        grad_j.transpose(-1, -2) @ fd_3
        + grad_a.transpose(-1, -2) @ fd_2
        + grad_v.transpose(-1, -2) @ fd_1
    ).transpose(-1, -2)
    # u_grad = (n_mask).transpose(-1, -2) @ p_grad

    u_grad = (p_grad.transpose(-1, -2) @ n_mask).transpose(-1, -2).contiguous()
    return u_grad


class CliqueTensorStep(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u_act,
        start_position,
        mask,
        n_mask,
        fd_1,
        fd_2,
        fd_3,
    ):
        state_position, state_vel, state_acc, state_jerk = jit_tensor_step_pos_clique(
            u_act, start_position, mask, n_mask, fd_1, fd_2, fd_3
        )
        ctx.save_for_backward(n_mask, fd_1, fd_2, fd_3)
        return state_position, state_vel, state_acc, state_jerk

    @staticmethod
    def backward(ctx, grad_out_p, grad_out_v, grad_out_a, grad_out_j):
        u_grad = None
        (n_mask, fd_1, fd_2, fd_3) = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            u_grad = jit_backward_pos_clique(
                grad_out_p, grad_out_v, grad_out_a, grad_out_j, n_mask, fd_1, fd_2, fd_3
            )
        return u_grad, None, None, None, None, None, None


class CliqueTensorStepKernel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u_act,
        start_position,
        start_velocity,
        start_acceleration,
        out_position,
        out_velocity,
        out_acceleration,
        out_jerk,
        traj_dt,
        out_grad_position,
    ):
        (
            state_position,
            state_velocity,
            state_acceleration,
            state_jerk,
        ) = tensor_step_pos_clique_fwd(
            out_position,
            out_velocity,
            out_acceleration,
            out_jerk,
            u_act,
            start_position,
            start_velocity,
            start_acceleration,
            traj_dt,
            out_position.shape[0],
            out_position.shape[1],
            out_position.shape[-1],
        )
        ctx.save_for_backward(traj_dt, out_grad_position)
        return state_position, state_velocity, state_acceleration, state_jerk

    @staticmethod
    def backward(ctx, grad_out_p, grad_out_v, grad_out_a, grad_out_j):
        u_grad = None

        if ctx.needs_input_grad[0]:
            (traj_dt, out_grad_position) = ctx.saved_tensors

            u_grad = tensor_step_pos_clique_bwd(
                out_grad_position,
                grad_out_p,
                grad_out_v,
                grad_out_a,
                grad_out_j,
                traj_dt,
                grad_out_p.shape[0],
                grad_out_p.shape[1],
                grad_out_p.shape[2],
            )
        return (
            u_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class CliqueTensorStepIdxKernel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u_act,
        start_position,
        start_velocity,
        start_acceleration,
        start_idx,
        out_position,
        out_velocity,
        out_acceleration,
        out_jerk,
        traj_dt,
        out_grad_position,
    ):
        (
            state_position,
            state_velocity,
            state_acceleration,
            state_jerk,
        ) = tensor_step_pos_clique_idx_fwd(
            out_position,
            out_velocity,
            out_acceleration,
            out_jerk,
            u_act,
            start_position,
            start_velocity,
            start_acceleration,
            start_idx,
            traj_dt,
            out_position.shape[0],
            out_position.shape[1],
            out_position.shape[-1],
        )

        ctx.save_for_backward(traj_dt, out_grad_position)
        return state_position, state_velocity, state_acceleration, state_jerk

    @staticmethod
    def backward(ctx, grad_out_p, grad_out_v, grad_out_a, grad_out_j):
        u_grad = None

        if ctx.needs_input_grad[0]:
            (traj_dt, out_grad_position) = ctx.saved_tensors

            u_grad = tensor_step_pos_clique_bwd(
                out_grad_position,
                grad_out_p,
                grad_out_v,
                grad_out_a,
                grad_out_j,
                traj_dt,
                grad_out_p.shape[0],
                grad_out_p.shape[1],
                grad_out_p.shape[2],
            )
        return (
            u_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )  # , None, None, None, None,None


class CliqueTensorStepCentralDifferenceKernel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u_act,
        start_position,
        start_velocity,
        start_acceleration,
        out_position,
        out_velocity,
        out_acceleration,
        out_jerk,
        traj_dt,
        out_grad_position,
    ):
        (
            state_position,
            state_velocity,
            state_acceleration,
            state_jerk,
        ) = tensor_step_pos_clique_fwd(
            out_position,
            out_velocity,
            out_acceleration,
            out_jerk,
            u_act,
            start_position,
            start_velocity,
            start_acceleration,
            traj_dt,
            out_position.shape[0],
            out_position.shape[1],
            out_position.shape[-1],
            0,
        )
        ctx.save_for_backward(traj_dt, out_grad_position)
        return state_position, state_velocity, state_acceleration, state_jerk

    @staticmethod
    def backward(ctx, grad_out_p, grad_out_v, grad_out_a, grad_out_j):
        u_grad = None

        if ctx.needs_input_grad[0]:
            (traj_dt, out_grad_position) = ctx.saved_tensors

            u_grad = tensor_step_pos_clique_bwd(
                out_grad_position,
                grad_out_p,
                grad_out_v,
                grad_out_a.contiguous(),
                grad_out_j.contiguous(),
                traj_dt,
                grad_out_p.shape[0],
                grad_out_p.shape[1],
                grad_out_p.shape[2],
                0,
            )
        return (
            u_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class CliqueTensorStepIdxCentralDifferenceKernel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u_act,
        start_position,
        start_velocity,
        start_acceleration,
        start_idx,
        out_position,
        out_velocity,
        out_acceleration,
        out_jerk,
        traj_dt,
        out_grad_position,
    ):
        (
            state_position,
            state_velocity,
            state_acceleration,
            state_jerk,
        ) = tensor_step_pos_clique_idx_fwd(
            out_position,
            out_velocity,
            out_acceleration,
            out_jerk,
            u_act,
            start_position,
            start_velocity,
            start_acceleration,
            start_idx.contiguous(),
            traj_dt,
            out_position.shape[0],
            out_position.shape[1],
            out_position.shape[-1],
            0,
        )

        ctx.save_for_backward(traj_dt, out_grad_position)
        return state_position, state_velocity, state_acceleration, state_jerk

    @staticmethod
    def backward(ctx, grad_out_p, grad_out_v, grad_out_a, grad_out_j):
        u_grad = None

        if ctx.needs_input_grad[0]:
            (traj_dt, out_grad_position) = ctx.saved_tensors

            u_grad = tensor_step_pos_clique_bwd(
                out_grad_position,
                grad_out_p,
                grad_out_v,
                grad_out_a.contiguous(),
                grad_out_j.contiguous(),
                traj_dt,
                grad_out_p.shape[0],
                grad_out_p.shape[1],
                grad_out_p.shape[2],
                0,
            )
        return (
            u_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )  # , None, None, None, None,None


class CliqueTensorStepCoalesceKernel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u_act,
        start_position,
        start_velocity,
        start_acceleration,
        out_position,
        out_velocity,
        out_acceleration,
        out_jerk,
        traj_dt,
        out_grad_position,
    ):
        state_position, state_velocity, state_acceleration, state_jerk = tensor_step_pos_clique_fwd(
            out_position.transpose(-1, -2).contiguous(),
            out_velocity.transpose(-1, -2).contiguous(),
            out_acceleration.transpose(-1, -2).contiguous(),
            out_jerk.transpose(-1, -2).contiguous(),
            u_act.transpose(-1, -2).contiguous(),
            start_position,
            start_velocity,
            start_acceleration,
            traj_dt,
            out_position.shape[0],
            out_position.shape[1],
            out_position.shape[-1],
        )
        ctx.save_for_backward(traj_dt, out_grad_position)
        return (
            state_position.transpose(-1, -2).contiguous(),
            state_velocity.transpose(-1, -2).contiguous(),
            state_acceleration.transpose(-1, -2).contiguous(),
            state_jerk.transpose(-1, -2).contiguous(),
        )

    @staticmethod
    def backward(ctx, grad_out_p, grad_out_v, grad_out_a, grad_out_j):
        u_grad = None
        (traj_dt, out_grad_position) = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            u_grad = tensor_step_pos_clique_bwd(
                out_grad_position.transpose(-1, -2).contiguous(),
                grad_out_p.transpose(-1, -2).contiguous(),
                grad_out_v.transpose(-1, -2).contiguous(),
                grad_out_a.transpose(-1, -2).contiguous(),
                grad_out_j.transpose(-1, -2).contiguous(),
                traj_dt,
                grad_out_p.shape[0],
                grad_out_p.shape[1],
                grad_out_p.shape[2],
            )
        return (
            u_grad.transpose(-1, -2).contiguous(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class AccelerationTensorStepKernel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u_act,
        start_position,
        start_velocity,
        start_acceleration,
        out_position,
        out_velocity,
        out_acceleration,
        out_jerk,
        traj_dt,
        out_grad_position,
    ):
        state_position, state_velocity, state_acceleration, state_jerk = tensor_step_acc_fwd(
            out_position,
            out_velocity,
            out_acceleration,
            out_jerk,
            u_act,
            start_position,
            start_velocity,
            start_acceleration,
            traj_dt,
            out_position.shape[0],
            out_position.shape[1],
            out_position.shape[-1],
        )
        ctx.save_for_backward(traj_dt, out_grad_position)
        return state_position, state_velocity, state_acceleration, state_jerk

    @staticmethod
    def backward(ctx, grad_out_p, grad_out_v, grad_out_a, grad_out_j):
        u_grad = None
        (traj_dt, out_grad_position) = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            raise NotImplementedError()
            u_grad = tensor_step_pos_clique_bwd(
                out_grad_position,
                grad_out_p,
                grad_out_v,
                grad_out_a,
                grad_out_j,
                traj_dt,
                out_grad_position.shape[0],
                out_grad_position.shape[1],
                out_grad_position.shape[2],
            )
        return u_grad, None, None, None, None, None, None, None, None, None


class AccelerationTensorStepIdxKernel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u_act,
        start_position,
        start_velocity,
        start_acceleration,
        start_idx,
        out_position,
        out_velocity,
        out_acceleration,
        out_jerk,
        traj_dt,
        out_grad_position,
    ):
        state_position, state_velocity, state_acceleration, state_jerk = tensor_step_acc_idx_fwd(
            out_position,
            out_velocity,
            out_acceleration,
            out_jerk,
            u_act,
            start_position,
            start_velocity,
            start_acceleration,
            start_idx,
            traj_dt,
            out_position.shape[0],
            out_position.shape[1],
            out_position.shape[-1],
        )
        ctx.save_for_backward(traj_dt, out_grad_position)
        return state_position, state_velocity, state_acceleration, state_jerk

    @staticmethod
    def backward(ctx, grad_out_p, grad_out_v, grad_out_a, grad_out_j):
        u_grad = None
        (traj_dt, out_grad_position) = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            raise NotImplementedError()
            u_grad = tensor_step_pos_clique_bwd(
                out_grad_position,
                grad_out_p,
                grad_out_v,
                grad_out_a,
                grad_out_j,
                traj_dt,
                out_grad_position.shape[0],
                out_grad_position.shape[1],
                out_grad_position.shape[2],
            )
        return u_grad, None, None, None, None, None, None, None, None, None, None


# @get_torch_jit_decorator()
def tensor_step_pos_clique(
    state: JointState,
    act: torch.Tensor,
    state_seq: JointState,
    mask_matrix: List[torch.Tensor],
    fd_matrix: List[torch.Tensor],
):
    (
        state_seq.position,
        state_seq.velocity,
        state_seq.acceleration,
        state_seq.jerk,
    ) = CliqueTensorStep.apply(
        act,
        state.position,
        mask_matrix[0],
        mask_matrix[1],
        fd_matrix[0],
        fd_matrix[1],
        fd_matrix[2],
    )
    return state_seq


def step_acc_semi_euler(state, act, diag_dt, n_dofs, integrate_matrix):
    q = state[..., :n_dofs]
    qd = state[..., n_dofs : 2 * n_dofs]
    qdd_new = act
    # diag_dt = torch.diag(dt_h)
    qd_new = euler_integrate(qd, qdd_new, diag_dt, integrate_matrix)
    q_new = euler_integrate(q, qd_new, diag_dt, integrate_matrix)
    state_seq = torch.cat((q_new, qd_new, qdd_new), dim=-1)
    return state_seq


# @get_torch_jit_decorator()
def tensor_step_acc_semi_euler(
    state, act, state_seq, diag_dt, integrate_matrix, integrate_matrix_pos
):
    # type: (Tensor, Tensor, Tensor, int, Tensor, Optional[Tensor]) -> Tensor
    # This is batch,n_dof
    state = state.unsqueeze(1)
    q = state.position  # [..., :n_dofs]
    qd = state.velocity  # [..., n_dofs : 2 * n_dofs]
    qdd_new = act
    # diag_dt = torch.diag(dt_h)
    qd_new = euler_integrate(qd, qdd_new, diag_dt, integrate_matrix)
    q_new = euler_integrate(q, qd_new, diag_dt, integrate_matrix_pos)
    state_seq.acceleration = qdd_new
    state_seq.velocity = qd_new
    state_seq.position = q_new

    return state_seq


# @get_torch_jit_decorator()
def tensor_step_vel(state, act, state_seq, dt_h, n_dofs, integrate_matrix, fd_matrix):
    # type: (Tensor, Tensor, Tensor, Tensor, int, Tensor, Tensor) -> Tensor

    # This is batch,n_dof
    state_seq[:, 0:1, : n_dofs * 3] = state
    q = state[..., :n_dofs]

    qd_new = act[:, :-1, :]
    # integrate velocities:
    dt_diag = torch.diag(dt_h)
    state_seq[:, 1:, n_dofs : n_dofs * 2] = qd_new
    qd = state_seq[:, :, n_dofs : n_dofs * 2]

    q_new = euler_integrate(q, qd, dt_diag, integrate_matrix)

    state_seq[:, :, :n_dofs] = q_new

    qdd = (torch.diag(1 / dt_h)) @ fd_matrix @ qd
    state_seq[:, 1:, n_dofs * 2 : n_dofs * 3] = qdd[:, :-1, :]

    return state_seq


# @get_torch_jit_decorator()
def tensor_step_pos(state, act, state_seq, fd_matrix):
    # This is batch,n_dof
    state_seq.position[:, 0, :] = state.position
    state_seq.velocity[:, 0, :] = state.velocity
    state_seq.acceleration[:, 0, :] = state.acceleration

    # integrate velocities:
    state_seq.position[:, 1:] = act[:, :-1, :]

    qd = fd_matrix @ state_seq.position  # [:, :, :n_dofs]
    state_seq.velocity[:, 1:] = qd[:, :-1, :]  # qd_new
    qdd = fd_matrix @ state_seq.velocity  # [:, :, n_dofs : n_dofs * 2]

    state_seq.acceleration[:, 1:] = qdd[:, :-1, :]
    # jerk = fd_matrix @ state_seq.acceleration

    return state_seq


# @get_torch_jit_decorator()
def tensor_step_pos_ik(act, state_seq):
    state_seq.position = act
    return state_seq


def tensor_linspace(start_tensor, end_tensor, steps=10):
    dist = end_tensor - start_tensor
    interpolate_matrix = (
        torch.ones((steps), device=start_tensor.device, dtype=start_tensor.dtype) / steps
    )
    cum_matrix = torch.cumsum(interpolate_matrix, dim=0)

    interp_tensor = start_tensor + cum_matrix * dist

    return interp_tensor


def sum_matrix(h, int_steps, tensor_args):
    sum_mat = torch.zeros(((h - 1) * int_steps, h), **(tensor_args.as_torch_dict()))
    for i in range(h - 1):
        sum_mat[i * int_steps : i * int_steps + int_steps, i] = 1.0
    # hack:
    # sum_mat[-1, -1] = 1.0
    return sum_mat


def interpolate_kernel(h, int_steps, tensor_args: TensorDeviceType):
    mat = torch.zeros(
        ((h - 1) * (int_steps), h), device=tensor_args.device, dtype=tensor_args.dtype
    )
    delta = torch.arange(0, int_steps, device=tensor_args.device, dtype=tensor_args.dtype) / (
        int_steps - 1
    )
    for i in range(h - 1):
        mat[i * int_steps : i * int_steps + int_steps, i] = delta.flip(0)
        mat[i * int_steps : i * int_steps + int_steps, i + 1] = delta
    return mat


def action_interpolate_kernel(h, int_steps, tensor_args: TensorDeviceType, offset: int = 4):
    mat = torch.zeros(
        ((h - 1) * (int_steps), h), device=tensor_args.device, dtype=tensor_args.dtype
    )
    delta = torch.arange(
        0, int_steps - offset + 1, device=tensor_args.device, dtype=tensor_args.dtype
    ) / (int_steps - offset)
    for i in range(h - 1):
        mat[i * int_steps : i * (int_steps) + int_steps - offset, i] = delta.flip(0)[1:]
        mat[i * int_steps : i * (int_steps) + int_steps - offset, i + 1] = delta[1:]
    mat[-offset:, 1] = 1.0

    return mat
