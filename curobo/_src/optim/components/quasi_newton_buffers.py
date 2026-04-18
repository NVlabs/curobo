# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Limited-memory ring buffers for quasi-Newton step direction computation.

Stores the (s, y, rho) history pairs plus initial position and gradient snapshots
(x_0, grad_0) used by both L-BFGS two-loop recursion and SR1 rank-1 updates.
"""

from typing import Optional

import torch

from curobo._src.optim.gradient.lbfgs_jit_helpers import (
    jit_lbfgs_update_buffers,
    lbfgs_shift_buffers_jit,
)
from curobo._src.types.device_cfg import DeviceCfg


class QuasiNewtonBuffers:
    """Manages limited-memory quasi-Newton history buffers."""

    def __init__(self, device_cfg: DeviceCfg, history: int):
        self.device_cfg = device_cfg
        self.history = history
        self.s: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self.rho: Optional[torch.Tensor] = None
        self.x_0: Optional[torch.Tensor] = None
        self.grad_0: Optional[torch.Tensor] = None
        self.alpha: Optional[torch.Tensor] = None
        self.step_q_buffer: Optional[torch.Tensor] = None

    def resize(self, num_problems: int, opt_dim: int):
        """Allocate or reallocate all buffers for given problem count and optimization dimension."""
        b = num_problems
        device = self.device_cfg.device
        dtype = self.device_cfg.dtype

        self.x_0 = torch.zeros((b, opt_dim, 1), device=device, dtype=dtype)
        self.grad_0 = torch.zeros((b, opt_dim, 1), device=device, dtype=dtype)

        self.y = torch.zeros(
            (self.history, b, opt_dim, 1), device=device, dtype=dtype
        )
        self.s = torch.zeros(
            (self.history, b, opt_dim, 1), device=device, dtype=dtype
        )
        self.rho = torch.zeros(
            (self.history, b, 1, 1), device=device, dtype=dtype
        )

        # LBFGS two-loop scratch space (also allocated for SR1 to keep interface uniform)
        self.step_q_buffer = torch.zeros((b, opt_dim), device=device, dtype=dtype)
        self.alpha = torch.zeros(
            (self.history, b, 1, 1), device=device, dtype=dtype
        )

    def clear(self, mask: Optional[torch.Tensor] = None):
        """Clear history buffers. CUDA graph compatible.

        Args:
            mask: Boolean mask [num_problems]. If None, clear all.
        """
        if mask is None:
            self.s.fill_(0.0)
            self.y.fill_(0.0)
            self.rho.fill_(0.0)
            self.alpha.fill_(0.0)
            self.step_q_buffer.fill_(0.0)
        else:
            self.s[:, mask] = 0.0
            self.y[:, mask] = 0.0
            self.rho[:, mask] = 0.0
            self.alpha[:, mask] = 0.0
            self.step_q_buffer[mask] = 0.0

    def set_reference(
        self,
        x: torch.Tensor,
        grad: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """Set the quasi-Newton reference point (x_0, grad_0).

        Args:
            x: Reference position [batch, opt_dim, 1].
            grad: Reference gradient [batch, opt_dim, 1].
            mask: Optional boolean mask for selective update.
        """
        if mask is None:
            self.x_0.copy_(x)
            self.grad_0.copy_(grad)
        else:
            new_x = torch.where(
                mask.unsqueeze(-1).unsqueeze(-1).expand_as(self.x_0), x, self.x_0
            )
            new_grad = torch.where(
                mask.unsqueeze(-1).unsqueeze(-1).expand_as(self.grad_0), grad, self.grad_0
            )
            self.x_0.copy_(new_x)
            self.grad_0.copy_(new_grad)

    def update(self, q: torch.Tensor, grad_q: torch.Tensor):
        """Update (s, y, rho) ring buffers and reference point.

        Args:
            q: Current position [batch, opt_dim].
            grad_q: Current gradient [batch, 1, opt_dim].
        """
        (
            self.s,
            self.y,
            self.rho,
            self.x_0,
            self.grad_0,
        ) = jit_lbfgs_update_buffers(
            q,
            grad_q,
            self.s,
            self.y,
            self.rho,
            self.x_0,
            self.grad_0,
            True,  # stable_mode always True
        )

    def shift(self, shift_steps: int, action_dim: int):
        """Shift buffers for MPC warm start.

        Args:
            shift_steps: Number of timesteps to shift.
            action_dim: Action dimensionality (needed for reshape).
        """
        self.x_0, self.grad_0, self.y, self.s = lbfgs_shift_buffers_jit(
            self.x_0,
            self.grad_0,
            self.y,
            self.s,
            shift_steps,
            action_dim,
        )
