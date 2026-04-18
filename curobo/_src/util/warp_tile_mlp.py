# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp Tile-based MLP with PyTorch autograd support.

Uses Warp's tile API for efficient matrix operations with shared memory.
Requires biases as 2D arrays (hidden_dim, 1) for tile_broadcast compatibility.

Reference:
    https://github.com/NVIDIA/warp/blob/main/warp/examples/tile/example_tile_mlp.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import warp as wp

from curobo._src.curobolib.cuda_ops.tensor_checks import check_float32_tensors

from curobo._src.util.warp import get_warp_device_stream


@dataclass
class WarpTileMLPTensors:
    """Pre-allocated tensors for WarpTileMLP forward and backward passes."""

    output: torch.Tensor  # [batch_size, n_out]
    grad_q: torch.Tensor  # [batch_size, n_input]
    grad_w0: torch.Tensor  # [hidden_dim, n_input]
    grad_b0: torch.Tensor  # [hidden_dim, 1]
    grad_w1: torch.Tensor  # [hidden_dim, hidden_dim]
    grad_b1: torch.Tensor  # [hidden_dim, 1]
    grad_w2: torch.Tensor  # [n_out, hidden_dim]
    grad_b2: torch.Tensor  # [n_out, 1]
    _batch_size: Optional[int] = None

    def __post_init__(self):
        self._batch_size = self.output.shape[0]

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def zero_grad(self):
        """Zero out all gradient tensors before backward pass."""
        self.grad_q.zero_()
        self.grad_w0.zero_()
        self.grad_b0.zero_()
        self.grad_w1.zero_()
        self.grad_b1.zero_()
        self.grad_w2.zero_()
        self.grad_b2.zero_()


class WarpTileMLPFunction(torch.autograd.Function):
    """PyTorch autograd function wrapping Warp tile MLP."""

    @staticmethod
    def forward(ctx, q, w0, b0, w1, b1, w2, b2, kernel, tile_threads, batch_tensors):
        batch_size = q.shape[0]
        device = q.device

        # Use pre-allocated output tensor
        output = batch_tensors.output

        wp.launch_tiled(
            kernel=kernel,
            dim=[batch_size],
            inputs=[
                wp.from_torch(q.detach(), dtype=wp.float32),
                wp.from_torch(w0.detach(), dtype=wp.float32),
                wp.from_torch(b0.detach(), dtype=wp.float32),
                wp.from_torch(w1.detach(), dtype=wp.float32),
                wp.from_torch(b1.detach(), dtype=wp.float32),
                wp.from_torch(w2.detach(), dtype=wp.float32),
                wp.from_torch(b2.detach(), dtype=wp.float32),
            ],
            outputs=[wp.from_torch(output.detach(), dtype=wp.float32)],
            block_dim=tile_threads,
            device=get_warp_device_stream(q)[0],
            stream=get_warp_device_stream(q)[1],
        )
        ctx.save_for_backward(q, w0, b0, w1, b1, w2, b2, output)
        ctx.kernel = kernel
        ctx.tile_threads = tile_threads
        ctx.batch_tensors = batch_tensors

        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, w0, b0, w1, b1, w2, b2, output = ctx.saved_tensors
        kernel = ctx.kernel
        tile_threads = ctx.tile_threads
        batch_tensors = ctx.batch_tensors

        # Zero and use pre-allocated gradient tensors
        batch_tensors.zero_grad()
        grad_q = batch_tensors.grad_q
        grad_w0 = batch_tensors.grad_w0
        grad_b0 = batch_tensors.grad_b0
        grad_w1 = batch_tensors.grad_w1
        grad_b1 = batch_tensors.grad_b1
        grad_w2 = batch_tensors.grad_w2
        grad_b2 = batch_tensors.grad_b2

        batch_size = q.shape[0]
        device = q.device
        check_float32_tensors(
            device,
            grad_output=grad_output,
            q=q, w0=w0, b0=b0, w1=w1, b1=b1, w2=w2, b2=b2, output=output,
        )

        wp_grad_q = wp.from_torch(grad_q, dtype=wp.float32)
        wp_grad_w0 = wp.from_torch(grad_w0, dtype=wp.float32)
        wp_grad_b0 = wp.from_torch(grad_b0, dtype=wp.float32)
        wp_grad_w1 = wp.from_torch(grad_w1, dtype=wp.float32)
        wp_grad_b1 = wp.from_torch(grad_b1, dtype=wp.float32)
        wp_grad_w2 = wp.from_torch(grad_w2, dtype=wp.float32)
        wp_grad_b2 = wp.from_torch(grad_b2, dtype=wp.float32)
        wp_grad_output = wp.from_torch(grad_output, dtype=wp.float32)

        # Create Warp arrays
        wp_q = wp.from_torch(q.detach(), dtype=wp.float32)
        wp_w0 = wp.from_torch(w0.detach(), dtype=wp.float32)
        wp_b0 = wp.from_torch(b0.detach(), dtype=wp.float32)
        wp_w1 = wp.from_torch(w1.detach(), dtype=wp.float32)
        wp_b1 = wp.from_torch(b1.detach(), dtype=wp.float32)
        wp_w2 = wp.from_torch(w2.detach(), dtype=wp.float32)
        wp_b2 = wp.from_torch(b2.detach(), dtype=wp.float32)
        wp_output = wp.from_torch(output.detach(), dtype=wp.float32)

        # Launch backward
        wp.launch_tiled(
            kernel=kernel,
            dim=[batch_size],
            inputs=[wp_q, wp_w0, wp_b0, wp_w1, wp_b1, wp_w2, wp_b2],
            outputs=[wp_output],
            adj_inputs=[
                wp_grad_q if ctx.needs_input_grad[0] else None,
                wp_grad_w0 if ctx.needs_input_grad[1] else None,
                wp_grad_b0 if ctx.needs_input_grad[2] else None,
                wp_grad_w1 if ctx.needs_input_grad[3] else None,
                wp_grad_b1 if ctx.needs_input_grad[4] else None,
                wp_grad_w2 if ctx.needs_input_grad[5] else None,
                wp_grad_b2 if ctx.needs_input_grad[6] else None,
            ],
            adj_outputs=[wp_grad_output],
            block_dim=tile_threads,
            device=get_warp_device_stream(q)[0],
            adjoint=True,
            stream=get_warp_device_stream(q)[1],
        )

        return (
            grad_q if ctx.needs_input_grad[0] else None,
            grad_w0 if ctx.needs_input_grad[1] else None,
            grad_b0 if ctx.needs_input_grad[2] else None,
            grad_w1 if ctx.needs_input_grad[3] else None,
            grad_b1 if ctx.needs_input_grad[4] else None,
            grad_w2 if ctx.needs_input_grad[5] else None,
            grad_b2 if ctx.needs_input_grad[6] else None,
            None,  # kernel
            None,  # tile_threads
            None,  # batch_tensors
        )


class WarpTileMLP(torch.nn.Module):
    """Tile-based Warp MLP.

    Uses Warp tile API for efficient shared memory matrix operations.
    Biases are stored as 2D tensors (hidden_dim, 1) for tile_broadcast.
    """

    def __init__(
        self, n_input: int = 7, n_out: int = 1, hidden_dim: int = 32, tile_threads: int = 32
    ):
        """Initialize the network.

        Args:
            n_input: Number of DOF (input dimension).
            n_out: Number of output frames (output dimension).
            hidden_dim: Hidden dimension.
            tile_threads: Number of threads per block for tiling.
        """
        super().__init__()

        self.n_input = n_input
        self.n_out = n_out
        self.hidden_dim = hidden_dim
        self._tile_threads = tile_threads
        self._batch_tensors: Optional[WarpTileMLPTensors] = None

        input_dim = n_input

        # Create the Warp kernel with compile-time constants
        self._tile_mlp_forward_kernel = WarpTileMLP.create_tile_mlp_kernel(
            n_input=n_input, hidden_dim=hidden_dim, n_out=n_out
        )

        # Initialize weights (Xavier)
        self.weights_0 = torch.nn.Parameter(
            torch.empty(hidden_dim, input_dim).uniform_(
                -1.0 / np.sqrt(input_dim), 1.0 / np.sqrt(input_dim)
            )
        )
        # 2D bias for tile_broadcast: shape (hidden_dim, 1)
        self.bias_0 = torch.nn.Parameter(torch.zeros(hidden_dim, 1))

        self.weights_1 = torch.nn.Parameter(
            torch.empty(hidden_dim, hidden_dim).uniform_(
                -1.0 / np.sqrt(hidden_dim), 1.0 / np.sqrt(hidden_dim)
            )
        )
        self.bias_1 = torch.nn.Parameter(torch.zeros(hidden_dim, 1))

        self.weights_2 = torch.nn.Parameter(
            torch.empty(n_out, hidden_dim).uniform_(
                -1.0 / np.sqrt(hidden_dim), 1.0 / np.sqrt(hidden_dim)
            )
        )
        self.bias_2 = torch.nn.Parameter(torch.zeros(n_out, 1))

    @property
    def tile_threads(self):
        return self._tile_threads

    @property
    def batch_tensors(self) -> Optional[WarpTileMLPTensors]:
        return self._batch_tensors

    def setup_batch_tensors(self, batch_size: int, device: torch.device) -> WarpTileMLPTensors:
        """Create pre-allocated tensors for forward and backward passes.

        Args:
            batch_size: Batch size for the tensors.
            device: Device to create tensors on.

        Returns:
            WarpTileMLPTensors containing pre-allocated output and gradient tensors.
        """
        self._batch_tensors = WarpTileMLPTensors(
            output=torch.zeros(batch_size, self.n_out, device=device, dtype=torch.float32),
            grad_q=torch.zeros(batch_size, self.n_input, device=device, dtype=torch.float32),
            grad_w0=torch.zeros(self.hidden_dim, self.n_input, device=device, dtype=torch.float32),
            grad_b0=torch.zeros(self.hidden_dim, 1, device=device, dtype=torch.float32),
            grad_w1=torch.zeros(
                self.hidden_dim, self.hidden_dim, device=device, dtype=torch.float32
            ),
            grad_b1=torch.zeros(self.hidden_dim, 1, device=device, dtype=torch.float32),
            grad_w2=torch.zeros(
                self.n_out, self.hidden_dim, device=device, dtype=torch.float32
            ),
            grad_b2=torch.zeros(self.n_out, 1, device=device, dtype=torch.float32),
        )
        return self._batch_tensors

    @staticmethod
    def create_tile_mlp_kernel(n_input: int, hidden_dim: int, n_out: int):
        """Create a Warp tile MLP kernel with compile-time constants.

        Args:
            n_input: Number of DOF (input dimension).
            hidden_dim: Hidden dimension.
            n_out: Number of output frames (output dimension).

        Returns:
            Compiled Warp kernel for tile-based MLP forward pass.
        """
        N_INPUT = wp.constant(n_input)
        DIM_HID = wp.constant(hidden_dim)
        N_OUT = wp.constant(n_out)

        @wp.func
        def activation_function(x: wp.float32) -> wp.float32:
            # use tanh:
            return wp.tanh(x)

        def _tile_mlp_forward_template(
            q: wp.array2d(dtype=wp.float32),  # [batch, n_input]
            weights_0: wp.array2d(dtype=wp.float32),  # [hidden_dim, input_dim]
            bias_0: wp.array2d(dtype=wp.float32),  # [hidden_dim, 1]
            weights_1: wp.array2d(dtype=wp.float32),  # [hidden_dim, hidden_dim]
            bias_1: wp.array2d(dtype=wp.float32),  # [hidden_dim, 1]
            weights_2: wp.array2d(dtype=wp.float32),  # [n_out, hidden_dim]
            bias_2: wp.array2d(dtype=wp.float32),  # [n_out, 1]
            output: wp.array2d(dtype=wp.float32),  # [batch, n_out]
        ):
            """Tile-based forward pass of MLP."""
            batch_idx = wp.tid()

            x = wp.tile_load(q, shape=(1, N_INPUT), offset=(batch_idx, 0))
            x_t = wp.tile_transpose(x)

            w0 = wp.tile_load(weights_0, shape=(DIM_HID, N_INPUT))
            b0 = wp.tile_load(bias_0, shape=(DIM_HID, 1))
            t0 = wp.tile_matmul(w0, x_t) + b0

            z0 = wp.tile_map(activation_function, t0)

            # Layer 1: Linear
            w1 = wp.tile_load(weights_1, shape=(DIM_HID, DIM_HID))
            b1 = wp.tile_load(bias_1, shape=(DIM_HID, 1))
            z1 = wp.tile_map(activation_function, wp.tile_matmul(w1, z0) + b1)

            # Layer 2: Linear, no activation
            w2 = wp.tile_load(weights_2, shape=(N_OUT, DIM_HID))
            b2 = wp.tile_load(bias_2, shape=(N_OUT, 1))
            out = wp.tile_matmul(w2, z1) + b2

            # Write output
            wp.tile_store(output, out, offset=(batch_idx, 0))

        _tile_mlp_forward_template.__name__ = f"_tile_mlp_forward_{n_input}_{hidden_dim}_{n_out}"
        _tile_mlp_forward_template.__qualname__ = (
            f"_tile_mlp_forward_{n_input}_{hidden_dim}_{n_out}"
        )

        return wp.kernel(enable_backward=True, module="unique")(_tile_mlp_forward_template)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Forward pass using Warp tile kernel.

        Args:
            q: Joint positions, shape [batch, n_input].

        Returns:
            Predicted output, shape [batch, n_out].

        Raises:
            RuntimeError: If setup_batch_tensors() was not called or batch size mismatch.
        """
        if self._batch_tensors is None:
            raise RuntimeError(
                "Batch tensors not initialized. Call setup_batch_tensors(batch_size, device) first."
            )
        if q.shape[0] != self._batch_tensors.batch_size:
            raise RuntimeError(
                f"Batch size mismatch: input has {q.shape[0]}, "
                f"but tensors were setup for {self._batch_tensors.batch_size}"
            )

        return WarpTileMLPFunction.apply(
            q,
            self.weights_0,
            self.bias_0,
            self.weights_1,
            self.bias_1,
            self.weights_2,
            self.bias_2,
            self._tile_mlp_forward_kernel,
            self._tile_threads,
            self._batch_tensors,
        )

    def forward_torch(self, q: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch forward for comparison."""
        x = torch.cat([q], dim=-1)
        z0 = torch.nn.functional.silu(x @ self.weights_0.T + self.bias_0.T)
        z1 = torch.nn.functional.silu(z0 @ self.weights_1.T + self.bias_1.T)
        out = z1 @ self.weights_2.T + self.bias_2.T
        return out

    def save(self, path: str, metadata: Optional[dict] = None):
        """Save model weights and configuration to disk.

        Args:
            path: Path to save the model file.
            metadata: Optional additional metadata to save with the model.
        """
        from pathlib import Path

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_type": "WarpTileMLP",
            "n_input": self.n_input,
            "n_out": self.n_out,
            "hidden_dim": self.hidden_dim,
            "tile_threads": self._tile_threads,
        }

        if metadata is not None:
            checkpoint["metadata"] = metadata

        torch.save(checkpoint, save_path)

    @classmethod
    def load(
        cls,
        path: str,
        device: str = "cuda:0",
        batch_size: int = 1,
    ) -> "WarpTileMLP":
        """Load a saved WarpTileMLP model from disk.

        Args:
            path: Path to the saved model file.
            device: Device to load the model on.
            batch_size: Batch size to setup tensors for.

        Returns:
            Loaded WarpTileMLP model ready for inference.
        """
        device_obj = torch.device(device)
        checkpoint = torch.load(path, map_location=device_obj, weights_only=False)

        model = cls(
            n_input=checkpoint["n_input"],
            n_out=checkpoint["n_out"],
            hidden_dim=checkpoint["hidden_dim"],
            tile_threads=checkpoint.get("tile_threads", checkpoint["hidden_dim"]),
        ).to(device_obj)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Setup batch tensors for inference
        model.setup_batch_tensors(batch_size=batch_size, device=device_obj)

        return model

    @staticmethod
    def get_metadata(path: str) -> dict:
        """Get metadata from a saved model file without loading the full model.

        Args:
            path: Path to the saved model file.

        Returns:
            Dictionary containing model configuration and optional metadata.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "n_input": checkpoint["n_input"],
            "n_out": checkpoint["n_out"],
            "hidden_dim": checkpoint["hidden_dim"],
            "tile_threads": checkpoint.get("tile_threads", checkpoint["hidden_dim"]),
            "metadata": checkpoint.get("metadata", {}),
        }

