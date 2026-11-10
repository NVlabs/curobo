# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Limited-memory ring buffers for the fused L-BFGS step direction.

Stores the (s, y, rho) history pairs plus initial position and gradient snapshots
(x_0, grad_0) used by the circular CUDA kernel.
"""

from typing import Optional

import torch

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.tensor_util import shift_buffer
from curobo._src.util.torch_util import get_torch_jit_decorator


@get_torch_jit_decorator()
def _shift_quasi_newton_buffers(
    x_0: torch.Tensor,
    grad_0: torch.Tensor,
    y_buffer: torch.Tensor,
    s_buffer: torch.Tensor,
    shift_steps: int,
    action_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shift L-BFGS state along the trajectory dimension for MPC warm starts."""
    shift_d = shift_steps * action_dim
    x_0.copy_(shift_buffer(x_0, shift_d, action_dim, shift_steps))
    grad_0.copy_(shift_buffer(grad_0, shift_d, action_dim, shift_steps))
    y_buffer.copy_(shift_buffer(y_buffer, shift_d, action_dim, shift_steps))
    s_buffer.copy_(shift_buffer(s_buffer, shift_d, action_dim, shift_steps))
    return x_0, grad_0, y_buffer, s_buffer


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
        self.step_q_buffer: Optional[torch.Tensor] = None
        self.history_head: Optional[torch.Tensor] = None
        self.history_count: Optional[torch.Tensor] = None
        self.compact_inverse_r: Optional[torch.Tensor] = None
        self.compact_matrix_workspace: Optional[torch.Tensor] = None
        self._compact_identity: Optional[torch.Tensor] = None

    def resize(
        self,
        num_problems: int,
        opt_dim: int,
        use_compact_operator: bool = False,
        compact_workspace_elements: int = 0,
    ) -> None:
        """Allocate or reallocate all buffers for given problem count and optimization dimension."""
        b = num_problems
        device = self.device_cfg.device
        dtype = self.device_cfg.dtype

        self.x_0 = torch.zeros((b, opt_dim, 1), device=device, dtype=dtype)
        self.grad_0 = torch.zeros((b, opt_dim, 1), device=device, dtype=dtype)

        self.y = torch.zeros((self.history, b, opt_dim, 1), device=device, dtype=dtype)
        self.s = torch.zeros((self.history, b, opt_dim, 1), device=device, dtype=dtype)
        self.rho = torch.zeros((self.history, b, 1, 1), device=device, dtype=dtype)

        # The fused kernel writes the final direction into this persistent workspace.
        self.step_q_buffer = torch.zeros((b, opt_dim), device=device, dtype=dtype)
        self.history_head = torch.zeros((b,), device=device, dtype=torch.int32)
        self.history_count = torch.zeros((b,), device=device, dtype=torch.int32)

        if use_compact_operator:
            self._compact_identity = torch.eye(self.history, device=device, dtype=dtype)
            self.compact_inverse_r = self._compact_identity.expand(b, -1, -1).clone()
            workspace_shape = (
                (b, compact_workspace_elements) if compact_workspace_elements > 0 else (1,)
            )
            self.compact_matrix_workspace = torch.empty(
                workspace_shape,
                device=device,
                dtype=dtype,
            )
            self.rho.fill_(1.0)
        else:
            self._compact_identity = None
            self.compact_inverse_r = None
            self.compact_matrix_workspace = None

    def clear(self, mask: Optional[torch.Tensor] = None) -> None:
        """Clear history buffers. CUDA graph compatible.

        Args:
            mask: Boolean mask [num_problems]. If None, clear all.
        """
        rho_reset_value = 1.0 if self.compact_inverse_r is not None else 0.0
        if mask is None:
            self.s.fill_(0.0)
            self.y.fill_(0.0)
            self.rho.fill_(rho_reset_value)
            self.step_q_buffer.fill_(0.0)
            self.history_head.fill_(0)
            self.history_count.fill_(0)
            if self.compact_inverse_r is not None:
                self.compact_inverse_r.copy_(
                    self._compact_identity.expand_as(self.compact_inverse_r)
                )
        else:
            history_mask = mask.view(1, -1, 1, 1)
            state_mask = mask.view(-1, 1)
            self.s.copy_(torch.where(history_mask, torch.zeros_like(self.s), self.s))
            self.y.copy_(torch.where(history_mask, torch.zeros_like(self.y), self.y))
            self.rho.copy_(
                torch.where(
                    history_mask,
                    torch.full_like(self.rho, rho_reset_value),
                    self.rho,
                )
            )
            self.step_q_buffer.copy_(
                torch.where(state_mask, torch.zeros_like(self.step_q_buffer), self.step_q_buffer)
            )
            self.history_head.copy_(
                torch.where(mask, torch.zeros_like(self.history_head), self.history_head)
            )
            self.history_count.copy_(
                torch.where(mask, torch.zeros_like(self.history_count), self.history_count)
            )
            if self.compact_inverse_r is not None:
                compact_mask = mask.view(-1, 1, 1)
                self.compact_inverse_r.copy_(
                    torch.where(
                        compact_mask,
                        self._compact_identity.expand_as(self.compact_inverse_r),
                        self.compact_inverse_r,
                    )
                )

    def rebuild_compact_state(self) -> None:
        """Rebuild compact inverse and curvature state after shifting history."""
        if self.compact_inverse_r is None:
            return

        self.compact_inverse_r.copy_(self._compact_identity.expand_as(self.compact_inverse_r))
        batch_size = self.history_count.shape[0]
        for batch_index in range(batch_size):
            count = int(self.history_count[batch_index].item())
            if count == 0:
                continue
            head = int(self.history_head[batch_index].item())
            logical_indices = (
                torch.arange(count, device=self.device_cfg.device) + head - count
            ) % self.history
            history_s = self.s[logical_indices, batch_index, :, 0]
            history_y = self.y[logical_indices, batch_index, :, 0]
            upper = torch.triu(history_s @ history_y.transpose(0, 1))
            inverse, info = torch.linalg.inv_ex(upper)
            if bool(info.item()) or not bool(torch.all(torch.isfinite(inverse)).item()):
                self.s[:, batch_index].zero_()
                self.y[:, batch_index].zero_()
                self.rho[:, batch_index].fill_(1.0)
                self.history_head[batch_index] = 0
                self.history_count[batch_index] = 0
                continue
            self.rho[:, batch_index].fill_(1.0)
            self.rho[logical_indices, batch_index, 0, 0] = 1.0 / torch.diagonal(upper)
            self.compact_inverse_r[batch_index][
                logical_indices[:, None], logical_indices[None, :]
            ] = inverse

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
            new_x = torch.where(mask.unsqueeze(-1).unsqueeze(-1).expand_as(self.x_0), x, self.x_0)
            new_grad = torch.where(
                mask.unsqueeze(-1).unsqueeze(-1).expand_as(self.grad_0), grad, self.grad_0
            )
            self.x_0.copy_(new_x)
            self.grad_0.copy_(new_grad)

    def shift(self, shift_steps: int, action_dim: int):
        """Shift buffers for MPC warm start.

        Args:
            shift_steps: Number of timesteps to shift.
            action_dim: Action dimensionality (needed for reshape).
        """
        self.x_0, self.grad_0, self.y, self.s = _shift_quasi_newton_buffers(
            self.x_0,
            self.grad_0,
            self.y,
            self.s,
            shift_steps,
            action_dim,
        )
