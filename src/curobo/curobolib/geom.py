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
# Third Party
import torch

# CuRobo
from curobo.util.logger import log_warn
from curobo.util.torch_utils import get_torch_jit_decorator

try:
    # CuRobo
    from curobo.curobolib import geom_cu

except ImportError:
    log_warn("geom_cu binary not found, jit compiling...")
    # Third Party
    from torch.utils.cpp_extension import load

    # CuRobo
    from curobo.util_file import add_cpp_path

    geom_cu = load(
        name="geom_cu",
        sources=add_cpp_path(
            [
                "geom_cuda.cpp",
                "sphere_obb_kernel.cu",
                "pose_distance_kernel.cu",
                "self_collision_kernel.cu",
            ]
        ),
    )


def get_self_collision_distance(
    out_distance,
    out_vec,
    sparse_index,
    robot_spheres,
    collision_offset,
    weight,
    coll_matrix,
    thread_locations,
    thread_size,
    b_size,
    nspheres,
    compute_grad,
    checks_per_thread=32,
    experimental_kernel=True,
):
    r = geom_cu.self_collision_distance(
        out_distance,
        out_vec,
        sparse_index,
        robot_spheres,
        collision_offset,
        weight,
        coll_matrix,
        thread_locations,
        thread_size,
        b_size,
        nspheres,
        compute_grad,
        checks_per_thread,
        experimental_kernel,
    )

    out_distance = r[0]
    out_vec = r[1]
    return out_distance, out_vec


class SelfCollisionDistance(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        out_distance,
        out_vec,
        sparse_idx,
        robot_spheres,
        sphere_offset,
        weight,
        coll_matrix,
        thread_locations,
        max_thread,
        checks_per_thread: int,
        experimental_kernel: bool,
        return_loss: bool = False,
    ):
        # get batch size
        b, h, n_spheres, _ = robot_spheres.shape
        out_distance, out_vec = get_self_collision_distance(
            out_distance,
            out_vec,
            sparse_idx,
            robot_spheres,  # .view(-1, 4),
            sphere_offset,
            weight,
            coll_matrix.view(-1),
            thread_locations,
            max_thread,
            b * h,
            n_spheres,
            robot_spheres.requires_grad,
            checks_per_thread,
            experimental_kernel,
        )
        ctx.return_loss = return_loss
        ctx.save_for_backward(out_vec)
        return out_distance

    @staticmethod
    def backward(ctx, grad_out_distance):
        sphere_grad = None
        if ctx.needs_input_grad[3]:
            (g_vec,) = ctx.saved_tensors
            if ctx.return_loss:
                g_vec = g_vec * grad_out_distance.view(*g_vec.shape[:2], 1, 1)
            sphere_grad = g_vec
        return None, None, None, sphere_grad, None, None, None, None, None, None, None, None


class SelfCollisionDistanceLoss(SelfCollisionDistance):
    @staticmethod
    def backward(ctx, grad_out_distance):
        sphere_grad = None
        if ctx.needs_input_grad[3]:
            (g_vec,) = ctx.saved_tensors
            sphere_grad = g_vec * grad_out_distance.unsqueeze(1)
        return None, None, None, sphere_grad, None, None, None, None, None, None, None


def get_pose_distance(
    out_distance,
    out_position_distance,
    out_rotation_distance,
    out_p_vec,
    out_q_vec,
    out_idx,
    current_position,
    goal_position,
    current_quat,
    goal_quat,
    vec_weight,
    weight,
    vec_convergence,
    run_weight,
    run_vec_weight,
    offset_waypoint,
    offset_tstep_fraction,
    batch_pose_idx,
    project_distance,
    batch_size,
    horizon,
    mode=1,
    num_goals=1,
    write_grad=False,
    write_distance=False,
    use_metric=False,
):
    if batch_pose_idx.shape[0] != batch_size:
        raise ValueError("Index buffer size is different from batch size")

    r = geom_cu.pose_distance(
        out_distance,
        out_position_distance,
        out_rotation_distance,
        out_p_vec,
        out_q_vec,
        out_idx,
        current_position,
        goal_position.view(-1),
        current_quat,
        goal_quat.view(-1),
        vec_weight,
        weight,
        vec_convergence,
        run_weight,
        run_vec_weight,
        offset_waypoint,
        offset_tstep_fraction,
        batch_pose_idx,
        project_distance,
        batch_size,
        horizon,
        mode,
        num_goals,
        write_grad,
        write_distance,
        use_metric,
    )

    out_distance = r[0]
    out_position_distance = r[1]
    out_rotation_distance = r[2]

    out_p_vec = r[3]
    out_q_vec = r[4]

    out_idx = r[5]
    return out_distance, out_position_distance, out_rotation_distance, out_p_vec, out_q_vec, out_idx


def get_pose_distance_backward(
    out_grad_p,
    out_grad_q,
    grad_distance,
    grad_p_distance,
    grad_q_distance,
    pose_weight,
    grad_p_vec,
    grad_q_vec,
    batch_size,
    use_distance=False,
):
    r = geom_cu.pose_distance_backward(
        out_grad_p,
        out_grad_q,
        grad_distance,
        grad_p_distance,
        grad_q_distance,
        pose_weight,
        grad_p_vec,
        grad_q_vec,
        batch_size,
        use_distance,
    )
    return r[0], r[1]


@get_torch_jit_decorator()
def backward_PoseError_jit(grad_g_dist, grad_out_distance, weight, g_vec):
    grad_vec = grad_g_dist + (grad_out_distance * weight)
    grad = 1.0 * (grad_vec).unsqueeze(-1) * g_vec
    return grad


# full method:
@get_torch_jit_decorator()
def backward_full_PoseError_jit(
    grad_out_distance, grad_g_dist, grad_r_err, p_w, q_w, g_vec_p, g_vec_q
):
    p_grad = (grad_g_dist + (grad_out_distance * p_w)).unsqueeze(-1) * g_vec_p
    q_grad = (grad_r_err + (grad_out_distance * q_w)).unsqueeze(-1) * g_vec_q
    # p_grad = ((grad_out_distance * p_w)).unsqueeze(-1) * g_vec_p
    # q_grad = ((grad_out_distance * q_w)).unsqueeze(-1) * g_vec_q

    return p_grad, q_grad


class PoseErrorDistance(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        current_position,
        goal_position,
        current_quat,
        goal_quat,
        vec_weight,
        weight,
        vec_convergence,
        run_weight,
        run_vec_weight,
        offset_waypoint,
        offset_tstep_fraction,
        batch_pose_idx,
        project_distance,
        out_distance,
        out_position_distance,
        out_rotation_distance,
        out_p_vec,
        out_r_vec,
        out_idx,
        out_p_grad,
        out_q_grad,
        batch_size,
        horizon,
        mode,  # =PoseErrorType.BATCH_GOAL.value,
        num_goals,
        use_metric,
    ):
        # out_distance = current_position[..., 0].detach().clone() * 0.0
        # out_position_distance = out_distance.detach().clone()
        # out_rotation_distance = out_distance.detach().clone()
        # out_vec = (
        #    torch.cat((current_position.detach().clone(), current_quat.detach().clone()), dim=-1)
        #    * 0.0
        # )
        # out_idx = out_distance.clone().to(dtype=torch.long)

        (
            out_distance,
            out_position_distance,
            out_rotation_distance,
            out_p_vec,
            out_r_vec,
            out_idx,
        ) = get_pose_distance(
            out_distance,
            out_position_distance,
            out_rotation_distance,
            out_p_vec,
            out_r_vec,
            out_idx,
            current_position.contiguous(),
            goal_position,
            current_quat.contiguous(),
            goal_quat,
            vec_weight,
            weight,
            vec_convergence,
            run_weight,
            run_vec_weight,
            offset_waypoint,
            offset_tstep_fraction,
            batch_pose_idx,
            project_distance,
            batch_size,
            horizon,
            mode,
            num_goals,
            current_position.requires_grad,
            True,
            use_metric,
        )
        ctx.save_for_backward(out_p_vec, out_r_vec, weight, out_p_grad, out_q_grad)
        return out_distance, out_position_distance, out_rotation_distance, out_idx  # .view(-1,1)

    @staticmethod
    def backward(ctx, grad_out_distance, grad_g_dist, grad_r_err, grad_out_idx):
        (g_vec_p, g_vec_q, weight, out_grad_p, out_grad_q) = ctx.saved_tensors
        pos_grad = None
        quat_grad = None
        batch_size = g_vec_p.shape[0] * g_vec_p.shape[1]
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[2]:
            pos_grad, quat_grad = get_pose_distance_backward(
                out_grad_p,
                out_grad_q,
                grad_out_distance.contiguous(),
                grad_g_dist.contiguous(),
                grad_r_err.contiguous(),
                weight,
                g_vec_p,
                g_vec_q,
                batch_size,
                use_distance=True,
            )

        elif ctx.needs_input_grad[0]:
            pos_grad = backward_PoseError_jit(grad_g_dist, grad_out_distance, weight[1], g_vec_p)

        elif ctx.needs_input_grad[2]:
            quat_grad = backward_PoseError_jit(grad_r_err, grad_out_distance, weight[0], g_vec_q)

        return (
            pos_grad,
            None,
            quat_grad,
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
            None,
            None,
            None,
        )


class PoseError(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        current_position: torch.Tensor,
        goal_position: torch.Tensor,
        current_quat: torch.Tensor,
        goal_quat,
        vec_weight,
        weight,
        vec_convergence,
        run_weight,
        run_vec_weight,
        offset_waypoint,
        offset_tstep_fraction,
        batch_pose_idx,
        project_distance,
        out_distance,
        out_position_distance,
        out_rotation_distance,
        out_p_vec,
        out_r_vec,
        out_idx,
        out_p_grad,
        out_q_grad,
        batch_size,
        horizon,
        mode,
        num_goals,
        use_metric,
        return_loss,
    ):
        """Compute error in pose

        _extended_summary_

        Args:
            ctx: _description_
            current_position: _description_
            goal_position: _description_
            current_quat: _description_
            goal_quat: _description_
            vec_weight: _description_
            weight: _description_
            vec_convergence: _description_
            run_weight: _description_
            run_vec_weight: _description_
            offset_waypoint: _description_
            offset_tstep_fraction: _description_
            batch_pose_idx: _description_
            out_distance: _description_
            out_position_distance: _description_
            out_rotation_distance: _description_
            out_p_vec: _description_
            out_r_vec: _description_
            out_idx: _description_
            out_p_grad: _description_
            out_q_grad: _description_
            batch_size: _description_
            horizon: _description_
            mode: _description_
            num_goals: _description_
            use_metric: _description_
            project_distance: _description_
            return_loss: _description_

        Returns:
            _description_
        """
        # out_distance = current_position[..., 0].detach().clone() * 0.0
        # out_position_distance = out_distance.detach().clone()
        # out_rotation_distance = out_distance.detach().clone()
        # out_vec = (
        #    torch.cat((current_position.detach().clone(), current_quat.detach().clone()), dim=-1)
        #    * 0.0
        # )
        # out_idx = out_distance.clone().to(dtype=torch.long)
        ctx.return_loss = return_loss
        (
            out_distance,
            out_position_distance,
            out_rotation_distance,
            out_p_vec,
            out_r_vec,
            out_idx,
        ) = get_pose_distance(
            out_distance,
            out_position_distance,
            out_rotation_distance,
            out_p_vec,
            out_r_vec,
            out_idx,
            current_position.contiguous(),
            goal_position,
            current_quat.contiguous(),
            goal_quat,
            vec_weight,
            weight,
            vec_convergence,
            run_weight,
            run_vec_weight,
            offset_waypoint,
            offset_tstep_fraction,
            batch_pose_idx,
            project_distance,
            batch_size,
            horizon,
            mode,
            num_goals,
            current_position.requires_grad,
            False,
            use_metric,
        )
        ctx.save_for_backward(out_p_vec, out_r_vec)
        return out_distance

    @staticmethod
    def backward(ctx, grad_out_distance):  # , grad_g_dist, grad_r_err, grad_out_idx):
        pos_grad = None
        quat_grad = None
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[2]:
            (g_vec_p, g_vec_q) = ctx.saved_tensors
            pos_grad = g_vec_p
            quat_grad = g_vec_q
            if ctx.return_loss:
                pos_grad = pos_grad * grad_out_distance.unsqueeze(1)
                quat_grad = quat_grad * grad_out_distance.unsqueeze(1)

        elif ctx.needs_input_grad[0]:
            (g_vec_p, g_vec_q) = ctx.saved_tensors

            pos_grad = g_vec_p
            if ctx.return_loss:
                pos_grad = pos_grad * grad_out_distance.unsqueeze(1)
        elif ctx.needs_input_grad[2]:
            (g_vec_p, g_vec_q) = ctx.saved_tensors

            quat_grad = g_vec_q
            if ctx.return_loss:
                quat_grad = quat_grad * grad_out_distance.unsqueeze(1)
        return (
            pos_grad,
            None,
            quat_grad,
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
            None,
            None,
            None,
            None,
        )


class SdfSphereOBB(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query_sphere,
        out_buffer,
        grad_out_buffer,
        sparsity_idx,
        weight,
        activation_distance,
        max_distance,
        box_accel,
        box_dims,
        box_pose,
        box_enable,
        n_env_obb,
        env_query_idx,
        max_nobs,
        batch_size,
        horizon,
        n_spheres,
        transform_back,
        compute_distance,
        use_batch_env,
        return_loss: bool = False,
        sum_collisions: bool = True,
        compute_esdf: bool = False,
    ):
        r = geom_cu.closest_point(
            query_sphere,
            out_buffer,
            grad_out_buffer,
            sparsity_idx,
            weight,
            activation_distance,
            max_distance,
            box_accel,
            box_dims,
            box_pose,
            box_enable,
            n_env_obb,
            env_query_idx,
            max_nobs,
            batch_size,
            horizon,
            n_spheres,
            transform_back,
            compute_distance,
            use_batch_env,
            sum_collisions,
            compute_esdf,
        )
        # r[1][r[1]!=r[1]] = 0.0
        ctx.compute_esdf = compute_esdf
        ctx.return_loss = return_loss
        ctx.save_for_backward(r[1])
        return r[0]

    @staticmethod
    def backward(ctx, grad_output):
        grad_pt = None
        if ctx.needs_input_grad[0]:
            # if ctx.compute_esdf:
            #    raise NotImplementedError("Gradients not implemented for compute_esdf=True")
            (r,) = ctx.saved_tensors
            if ctx.return_loss:
                r = r * grad_output.unsqueeze(-1)
            grad_pt = r
        return (
            grad_pt,
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
            None,
            None,
        )


class SdfSweptSphereOBB(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query_sphere,
        out_buffer,
        grad_out_buffer,
        sparsity_idx,
        weight,
        activation_distance,
        speed_dt,
        box_accel,
        box_dims,
        box_pose,
        box_enable,
        n_env_obb,
        env_query_idx,
        max_nobs,
        batch_size,
        horizon,
        n_spheres,
        sweep_steps,
        enable_speed_metric,
        transform_back,
        compute_distance,
        use_batch_env,
        return_loss: bool = False,
        sum_collisions: bool = True,
    ):
        r = geom_cu.swept_closest_point(
            query_sphere,
            out_buffer,
            grad_out_buffer,
            sparsity_idx,
            weight,
            activation_distance,
            speed_dt,
            box_accel,
            box_dims,
            box_pose,
            box_enable,
            n_env_obb,
            env_query_idx,
            max_nobs,
            batch_size,
            horizon,
            n_spheres,
            sweep_steps,
            enable_speed_metric,
            transform_back,
            compute_distance,
            use_batch_env,
            sum_collisions,
        )
        ctx.return_loss = return_loss
        ctx.save_for_backward(
            r[1],
        )
        return r[0]

    @staticmethod
    def backward(ctx, grad_output):
        grad_pt = None
        if ctx.needs_input_grad[0]:
            (r,) = ctx.saved_tensors
            if ctx.return_loss:
                r = r * grad_output.unsqueeze(-1)
            grad_pt = r
        return (
            grad_pt,
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
            None,
            None,
            None,
        )


class SdfSphereVoxel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query_sphere,
        out_buffer,
        grad_out_buffer,
        sparsity_idx,
        weight,
        activation_distance,
        max_distance,
        grid_features,
        grid_params,
        grid_pose,
        grid_enable,
        n_env_grid,
        env_query_idx,
        max_nobs,
        batch_size,
        horizon,
        n_spheres,
        transform_back,
        compute_distance,
        use_batch_env,
        return_loss: bool = False,
        sum_collisions: bool = True,
        compute_esdf: bool = False,
    ):

        r = geom_cu.closest_point_voxel(
            query_sphere,
            out_buffer,
            grad_out_buffer,
            sparsity_idx,
            weight,
            activation_distance,
            max_distance,
            grid_features,
            grid_params,
            grid_pose,
            grid_enable,
            n_env_grid,
            env_query_idx,
            max_nobs,
            batch_size,
            horizon,
            n_spheres,
            transform_back,
            compute_distance,
            use_batch_env,
            sum_collisions,
            compute_esdf,
        )
        ctx.compute_esdf = compute_esdf
        ctx.return_loss = return_loss
        ctx.save_for_backward(r[1])
        return r[0]

    @staticmethod
    def backward(ctx, grad_output):
        grad_pt = None
        if ctx.needs_input_grad[0]:
            # if ctx.compute_esdf:
            #    raise NotImplementedError("Gradients not implemented for compute_esdf=True")
            (r,) = ctx.saved_tensors
            if ctx.return_loss:
                r = r * grad_output.unsqueeze(-1)
            grad_pt = r
        return (
            grad_pt,
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
            None,
            None,
        )


class SdfSweptSphereVoxel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query_sphere,
        out_buffer,
        grad_out_buffer,
        sparsity_idx,
        weight,
        activation_distance,
        max_distance,
        speed_dt,
        grid_features,
        grid_params,
        grid_pose,
        grid_enable,
        n_env_grid,
        env_query_idx,
        max_nobs,
        batch_size,
        horizon,
        n_spheres,
        sweep_steps,
        enable_speed_metric,
        transform_back,
        compute_distance,
        use_batch_env,
        return_loss: bool = False,
        sum_collisions: bool = True,
    ):
        r = geom_cu.swept_closest_point_voxel(
            query_sphere,
            out_buffer,
            grad_out_buffer,
            sparsity_idx,
            weight,
            activation_distance,
            max_distance,
            speed_dt,
            grid_features,
            grid_params,
            grid_pose,
            grid_enable,
            n_env_grid,
            env_query_idx,
            max_nobs,
            batch_size,
            horizon,
            n_spheres,
            sweep_steps,
            enable_speed_metric,
            transform_back,
            compute_distance,
            use_batch_env,
            sum_collisions,
        )

        ctx.return_loss = return_loss
        ctx.save_for_backward(
            r[1],
        )
        return r[0]

    @staticmethod
    def backward(ctx, grad_output):
        grad_pt = None
        if ctx.needs_input_grad[0]:
            (r,) = ctx.saved_tensors
            if ctx.return_loss:
                r = r * grad_output.unsqueeze(-1)
            grad_pt = r
        return (
            grad_pt,
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
            None,
            None,
            None,
            None,
        )
