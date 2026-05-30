# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark inverse dynamics CUDA kernel times with torch.profiler."""

from __future__ import annotations

import argparse

import tabulate
import torch
from torch.profiler import ProfilerActivity, profile

from curobo._src.robot.dynamics.dynamics import Dynamics
from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.config_io import join_path, load_yaml
from curobo.content import get_robot_configs_path


DEFAULT_ROBOTS = ["franka.yml", "dual_ur10e.yml", "unitree_g1.yml"]
DEFAULT_BATCH_SIZES = [1, 64, 256, 1024]


def _load_kinematics(robot_file_name: str, device_cfg: DeviceCfg) -> Kinematics:
    robot_file = load_yaml(join_path(get_robot_configs_path(), robot_file_name))
    if "robot_cfg" in robot_file:
        robot_file = robot_file["robot_cfg"]
    robot_file["kinematics"]["collision_link_names"] = None
    robot_file["kinematics"]["lock_joints"] = {}
    return Kinematics(KinematicsCfg.from_data_dict(robot_file, device_cfg=device_cfg))


def _build_dynamics(kin: Kinematics, device_cfg: DeviceCfg) -> Dynamics:
    return Dynamics(
        DynamicsCfg(kinematics_config=kin.kinematics_config, device_cfg=device_cfg)
    )


def _make_inverse_dynamics_inputs(
    batch_size: int,
    dof: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.rand((batch_size, dof), device=device)
    qd = torch.randn((batch_size, dof), device=device)
    qdd = torch.randn((batch_size, dof), device=device)
    return q.contiguous(), qd.contiguous(), qdd.contiguous()


def _run_forward(
    dynamics: Dynamics,
    q: torch.Tensor,
    qd: torch.Tensor,
    qdd: torch.Tensor,
) -> torch.Tensor:
    joint_state = JointState(position=q, velocity=qd, acceleration=qdd)
    return dynamics.compute_inverse_dynamics(joint_state)


def _zero_tensor_grad(tensor: torch.Tensor) -> None:
    if tensor.grad is not None:
        tensor.grad.zero_()


def _run_forward_backward(
    dynamics: Dynamics,
    q: torch.Tensor,
    qd: torch.Tensor,
    qdd: torch.Tensor,
    grad_tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    _zero_tensor_grad(q)
    _zero_tensor_grad(qd)
    _zero_tensor_grad(qdd)
    tau = _run_forward(dynamics, q, qd, qdd)
    torch.autograd.backward(tau, grad_tensors=grad_tau)
    return tau, q.grad, qd.grad, qdd.grad


def _event_time_us(event, names: tuple[str, ...]) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _kernel_rows(prof, profile_iters: int) -> list[dict]:
    aggregated = {}
    for event in prof.events():
        if "CUDA" not in str(getattr(event, "device_type", "")):
            continue
        if "kernel" not in event.key.lower():
            continue
        self_cuda_us = _event_time_us(
            event, ("self_device_time_total", "self_cuda_time_total")
        )
        if self_cuda_us <= 0.0:
            continue
        cuda_total_us = _event_time_us(event, ("device_time_total", "cuda_time_total"))
        row = aggregated.setdefault(
            event.key,
            {
                "kernel": event.key,
                "calls": 0,
                "self_cuda_total_us": 0.0,
                "cuda_total_us": 0.0,
            },
        )
        row["calls"] += 1
        row["self_cuda_total_us"] += self_cuda_us
        row["cuda_total_us"] += cuda_total_us

    rows = []
    for row in aggregated.values():
        row["self_cuda_per_iter_us"] = row["self_cuda_total_us"] / profile_iters
        rows.append(row)
    rows.sort(key=lambda x: x["self_cuda_total_us"], reverse=True)
    return rows


def _profile_run(
    run_fn,
    device: torch.device,
    warmup_iters: int,
    profile_iters: int,
) -> list[dict]:
    for _ in range(warmup_iters):
        run_fn()
    torch.cuda.synchronize(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        acc_events=True,
    ) as prof:
        for _ in range(profile_iters):
            run_fn()
    torch.cuda.synchronize(device)

    return _kernel_rows(prof, profile_iters)


def _sum_kernel_time(rows: list[dict], profile_iters: int, kernel_name: str = "") -> float:
    total_us = 0.0
    for row in rows:
        if kernel_name and kernel_name not in row["kernel"].lower():
            continue
        total_us += row["self_cuda_total_us"]
    return total_us / profile_iters


def _profile_one(
    robot: str,
    batch_size: int,
    device_cfg: DeviceCfg,
    warmup_iters: int,
    profile_iters: int,
) -> dict:
    device = device_cfg.device
    kin = _load_kinematics(robot, device_cfg)
    dynamics = _build_dynamics(kin, device_cfg)
    dynamics.setup_batch_size(batch_size=batch_size)
    q, qd, qdd = _make_inverse_dynamics_inputs(batch_size, kin.get_dof(), device)

    forward_rows = _profile_run(
        lambda: _run_forward(dynamics, q, qd, qdd),
        device=device,
        warmup_iters=warmup_iters,
        profile_iters=profile_iters,
    )

    q = q.detach().requires_grad_(True)
    qd = qd.detach().requires_grad_(True)
    qdd = qdd.detach().requires_grad_(True)
    grad_tau = torch.ones_like(q)
    forward_backward_rows = _profile_run(
        lambda: _run_forward_backward(dynamics, q, qd, qdd, grad_tau),
        device=device,
        warmup_iters=warmup_iters,
        profile_iters=profile_iters,
    )

    forward_us = _sum_kernel_time(forward_rows, profile_iters)
    forward_backward_us = _sum_kernel_time(forward_backward_rows, profile_iters)
    rnea_forward_us = sum(
        row["self_cuda_total_us"]
        for row in forward_rows
        if "rnea_forward" in row["kernel"].lower()
    ) / profile_iters
    rnea_backward_us = _sum_kernel_time(
        forward_backward_rows, profile_iters, "rnea_backward"
    )

    return {
        "robot": robot,
        "batch_size": batch_size,
        "forward_kernel_us": round(forward_us, 3),
        "forward_rnea_us": round(rnea_forward_us, 3),
        "backward_rnea_us": round(rnea_backward_us, 3),
        "forward_backward_kernel_us": round(forward_backward_us, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", nargs="+", default=DEFAULT_ROBOTS)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--profile-iters", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.profile_iters < 1:
        raise ValueError("profile_iters must be greater than zero")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("inverse dynamics profiling requires a CUDA device")

    torch.manual_seed(2)
    device_cfg = DeviceCfg(device=device)

    _ = torch.zeros((1,), device=device)
    summary = []
    for robot in args.robots:
        for batch_size in args.batch_sizes:
            summary.append(
                _profile_one(
                    robot=robot,
                    batch_size=batch_size,
                    device_cfg=device_cfg,
                    warmup_iters=args.warmup_iters,
                    profile_iters=args.profile_iters,
                )
            )

    print(tabulate.tabulate(summary, headers="keys", tablefmt="grid"))


if __name__ == "__main__":
    main()
