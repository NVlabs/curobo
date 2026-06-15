# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Block checkpoint helpers for mapper TSDF storage.

This module owns the disk payload contract for compact mapper block
checkpoints. It intentionally does not mutate mapper storage; storage classes
own in-memory block-pool updates.
"""

from __future__ import annotations

import inspect
import math
from os import PathLike
from typing import Any, Dict, Optional, Tuple, Union

import torch

from curobo._src.perception.mapper.constants import (
    DEFAULT_HASH_LAYOUT,
    PY_HASH_EMPTY,
    PY_HASH_PRIME_X,
    PY_HASH_PRIME_Y,
    PY_HASH_PRIME_Z,
    PY_POSITIVE_MASK,
)
from curobo.logging import log_and_raise


BLOCK_CHECKPOINT_FORMAT = "curobo.mapper_blocks"
BLOCK_CHECKPOINT_SCHEMA_VERSION = 0.8

BLOCK_CHECKPOINT_KEYS = {
    "format",
    "schema_version",
    "block_metadata",
    "blocks",
}
BLOCK_METADATA_KEYS = {
    "voxel_size",
    "block_size",
    "truncation_distance",
    "grid_center",
    "grid_shape",
    "has_dynamic",
    "has_static",
    "feature_dim",
}


def build_block_metadata(tsdf) -> Dict[str, Any]:
    """Build block compatibility metadata from a ``BlockSparseTSDF``."""
    data = tsdf.data
    return {
        "voxel_size": float(data.voxel_size),
        "block_size": int(data.block_size),
        "truncation_distance": float(data.truncation_distance),
        "grid_center": [
            float(x) for x in data.origin.detach().to(device="cpu").flatten().tolist()
        ],
        "grid_shape": [int(x) for x in data.grid_shape],
        "has_dynamic": bool(data.has_dynamic),
        "has_static": bool(data.has_static),
        "feature_dim": int(data.feature_dim),
    }


def save_block_checkpoint(
    file_path: Union[str, PathLike[str]],
    block_metadata: Dict[str, Any],
    blocks: Dict[str, torch.Tensor],
) -> None:
    """Save a compact mapper block checkpoint."""
    checkpoint = {
        "format": BLOCK_CHECKPOINT_FORMAT,
        "schema_version": BLOCK_CHECKPOINT_SCHEMA_VERSION,
        "block_metadata": dict(block_metadata),
        "blocks": clone_blocks_to_cpu(blocks),
    }
    validate_block_checkpoint(checkpoint)
    torch.save(checkpoint, file_path)


def load_block_checkpoint(file_path: Union[str, PathLike[str]]) -> Dict[str, Any]:
    """Load and validate a compact mapper block checkpoint."""
    kwargs: Dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    checkpoint = torch.load(file_path, **kwargs)
    validate_block_checkpoint(checkpoint)
    return checkpoint


def clone_blocks_to_cpu(blocks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Clone a block payload to CPU tensors."""
    out: Dict[str, torch.Tensor] = {}
    for key, value in blocks.items():
        if not isinstance(value, torch.Tensor):
            log_and_raise(
                f"Block payload field {key!r} must be a torch.Tensor, got "
                f"{type(value).__name__}."
            )
        out[key] = value.detach().to(device="cpu").clone()
    return out


def clone_blocks(blocks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Clone block payload tensors without changing their device."""
    out: Dict[str, torch.Tensor] = {}
    for key, value in blocks.items():
        if not isinstance(value, torch.Tensor):
            log_and_raise(
                f"Block payload field {key!r} must be a torch.Tensor, got "
                f"{type(value).__name__}."
            )
        out[key] = value.detach().clone()
    return out


def validate_block_checkpoint(checkpoint: Any) -> None:
    """Validate the top-level checkpoint schema."""
    if not isinstance(checkpoint, dict):
        log_and_raise(
            f"Block checkpoint must be a dict, got {type(checkpoint).__name__}."
        )
    if checkpoint.get("format") != BLOCK_CHECKPOINT_FORMAT:
        log_and_raise(
            f"Unsupported block checkpoint format {checkpoint.get('format')!r}; "
            f"expected {BLOCK_CHECKPOINT_FORMAT!r}."
        )
    if checkpoint.get("schema_version") != BLOCK_CHECKPOINT_SCHEMA_VERSION:
        log_and_raise(
            "Unsupported block checkpoint schema version "
            f"{checkpoint.get('schema_version')!r}; expected "
            f"{BLOCK_CHECKPOINT_SCHEMA_VERSION}."
        )
    extra = set(checkpoint.keys()).difference(BLOCK_CHECKPOINT_KEYS)
    if extra:
        log_and_raise(f"Block checkpoint has unsupported fields: {sorted(extra)}.")
    if "block_metadata" not in checkpoint:
        log_and_raise("Block checkpoint is missing 'block_metadata'.")
    if "blocks" not in checkpoint:
        log_and_raise("Block checkpoint is missing 'blocks'.")
    validate_block_metadata(checkpoint["block_metadata"])
    validate_block_payload(checkpoint["blocks"], checkpoint["block_metadata"])


def validate_block_metadata(block_metadata: Any) -> None:
    """Validate source block metadata."""
    if not isinstance(block_metadata, dict):
        log_and_raise(
            f"block_metadata must be a dict, got {type(block_metadata).__name__}."
        )
    missing = BLOCK_METADATA_KEYS.difference(block_metadata.keys())
    if missing:
        log_and_raise(f"block_metadata missing required fields: {sorted(missing)}.")
    extra = set(block_metadata.keys()).difference(BLOCK_METADATA_KEYS)
    if extra:
        log_and_raise(f"block_metadata has unsupported fields: {sorted(extra)}.")

    require_positive_float(block_metadata, "voxel_size")
    require_positive_float(block_metadata, "truncation_distance")
    require_positive_int(block_metadata, "block_size")
    require_nonnegative_int(block_metadata, "feature_dim")
    require_bool(block_metadata, "has_dynamic")
    require_bool(block_metadata, "has_static")
    require_vec3(block_metadata, "grid_center")
    require_int3(block_metadata, "grid_shape")
    if not block_metadata["has_dynamic"] and not block_metadata["has_static"]:
        log_and_raise("block_metadata requires at least one source channel.")


def validate_block_payload(
    blocks: Any,
    block_metadata: Dict[str, Any],
) -> None:
    """Validate block payload tensor fields against source metadata."""
    if not isinstance(blocks, dict):
        log_and_raise(f"blocks must be a dict, got {type(blocks).__name__}.")
    if "active_block_coords" not in blocks:
        log_and_raise("blocks missing required field 'active_block_coords'.")

    coords = require_tensor(blocks, "active_block_coords", torch.int32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        log_and_raise(
            "active_block_coords must have shape (N, 3), got "
            f"{tuple(coords.shape)}."
        )
    n_blocks = int(coords.shape[0])
    block_size = int(block_metadata["block_size"])
    block_voxels = block_size**3
    has_dynamic = bool(block_metadata["has_dynamic"])
    has_static = bool(block_metadata["has_static"])
    feature_dim = int(block_metadata["feature_dim"])

    expected_keys = {"active_block_coords"}
    if has_dynamic:
        expected_keys.update(("block_data", "block_rgb"))
    if feature_dim > 0:
        expected_keys.update(("block_features", "block_feature_weight"))
    if has_static:
        expected_keys.add("static_block_data")
    extra = set(blocks.keys()).difference(expected_keys)
    if extra:
        log_and_raise(f"blocks has unsupported fields: {sorted(extra)}.")

    if has_dynamic:
        block_data = require_tensor(blocks, "block_data", torch.float16)
        require_shape(block_data, "block_data", (n_blocks, block_voxels, 2))
        block_rgb = require_tensor(blocks, "block_rgb", torch.float16)
        require_shape(block_rgb, "block_rgb", (n_blocks, 4))

    if feature_dim > 0:
        block_features = require_tensor(blocks, "block_features", torch.float16)
        require_shape(block_features, "block_features", (n_blocks, feature_dim))
        block_feature_weight = require_tensor(
            blocks, "block_feature_weight", torch.float16
        )
        require_shape(block_feature_weight, "block_feature_weight", (n_blocks,))

    if has_static:
        static_block_data = require_tensor(blocks, "static_block_data", torch.float16)
        require_shape(static_block_data, "static_block_data", (n_blocks, block_voxels))


def validate_block_metadata_for_target(
    block_metadata: Dict[str, Any],
    tsdf,
) -> None:
    """Validate source metadata against a target ``BlockSparseTSDF``."""
    validate_block_metadata(block_metadata)
    data = tsdf.data
    require_close(
        float(block_metadata["voxel_size"]),
        float(data.voxel_size),
        "voxel_size",
    )
    require_close(
        float(block_metadata["truncation_distance"]),
        float(data.truncation_distance),
        "truncation_distance",
    )
    if int(block_metadata["block_size"]) != int(data.block_size):
        log_and_raise(
            f"block_size mismatch: checkpoint={block_metadata['block_size']}, "
            f"target={data.block_size}."
        )
    if bool(block_metadata["has_dynamic"]) != bool(data.has_dynamic):
        log_and_raise(
            f"has_dynamic mismatch: checkpoint={block_metadata['has_dynamic']}, "
            f"target={data.has_dynamic}."
        )
    if bool(block_metadata["has_static"]) != bool(data.has_static):
        log_and_raise(
            f"has_static mismatch: checkpoint={block_metadata['has_static']}, "
            f"target={data.has_static}."
        )
    if int(block_metadata["feature_dim"]) != int(data.feature_dim):
        log_and_raise(
            f"feature_dim mismatch: checkpoint={block_metadata['feature_dim']}, "
            f"target={data.feature_dim}."
        )

    source_center = [float(x) for x in block_metadata["grid_center"]]
    target_center = [float(x) for x in data.origin.detach().to(device="cpu").tolist()]
    for i, (a, b) in enumerate(zip(source_center, target_center)):
        if not math.isclose(a, b, rel_tol=0.0, abs_tol=1.0e-6):
            log_and_raise(
                f"grid_center[{i}] mismatch: checkpoint={a}, target={b}."
            )


def ceil_div_positive(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def signed_int64_from_uint64(value: int) -> int:
    value = value & ((1 << 64) - 1)
    if value >= (1 << 63):
        return value - (1 << 64)
    return value


def pack_hash_entry_host(bx: int, by: int, bz: int, pool_idx: int) -> int:
    layout = DEFAULT_HASH_LAYOUT
    x_mask, y_mask, z_mask = layout.coord_masks_xyz
    x_bias, y_bias, z_bias = layout.coord_bias_xyz
    x = (int(bx) + x_bias) & x_mask
    y = (int(by) + y_bias) & y_mask
    z = (int(bz) + z_bias) & z_mask
    v = int(pool_idx) & layout.value_mask
    entry = (
        (x << layout.x_shift)
        | (y << layout.y_shift)
        | (z << layout.z_shift)
        | v
    )
    return signed_int64_from_uint64(entry)


def spatial_hash_host(bx: int, by: int, bz: int, capacity: int) -> int:
    h = (int(bx) * PY_HASH_PRIME_X) ^ (int(by) * PY_HASH_PRIME_Y) ^ (
        int(bz) * PY_HASH_PRIME_Z
    )
    h = h & PY_POSITIVE_MASK
    return int(h % int(capacity))


def validate_import_block_coords_unique(coords: torch.Tensor) -> None:
    if coords.shape[0] == 0:
        return
    unique_coords = torch.unique(coords, dim=0)
    if int(unique_coords.shape[0]) != int(coords.shape[0]):
        log_and_raise("Block import payload contains duplicate active_block_coords.")


def validate_import_block_coords_for_hash_layout(coords: torch.Tensor) -> None:
    if coords.shape[0] == 0:
        return
    mins = DEFAULT_HASH_LAYOUT.coord_min_xyz
    maxs = DEFAULT_HASH_LAYOUT.coord_max_xyz
    for axis, axis_name in enumerate(("x", "y", "z")):
        axis_coords = coords[:, axis]
        too_low = axis_coords < mins[axis]
        too_high = axis_coords > maxs[axis]
        if bool((too_low | too_high).any().item()):
            log_and_raise(
                f"Block coordinate {axis_name} is outside hash layout range "
                f"[{mins[axis]}, {maxs[axis]}]."
            )


def validate_import_block_coords_for_grid(
    coords: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    block_size: int,
) -> None:
    if coords.shape[0] == 0:
        return
    nz, ny, nx = (int(v) for v in grid_shape)
    blocks_x = ceil_div_positive(nx, block_size)
    blocks_y = ceil_div_positive(ny, block_size)
    blocks_z = ceil_div_positive(nz, block_size)
    offsets = (blocks_x // 2, blocks_y // 2, blocks_z // 2)
    limits = (blocks_x, blocks_y, blocks_z)
    for axis, axis_name in enumerate(("x", "y", "z")):
        grid_coord = coords[:, axis] + offsets[axis]
        outside = (grid_coord < 0) | (grid_coord >= limits[axis])
        if bool(outside.any().item()):
            valid_min = -offsets[axis]
            valid_max = limits[axis] - offsets[axis] - 1
            log_and_raise(
                f"Block coordinate {axis_name} is outside target grid range "
                f"[{valid_min}, {valid_max}]."
            )


def rebuild_import_hash_state(
    coords: torch.Tensor,
    hash_capacity: int,
    max_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hash_table = torch.full((hash_capacity,), PY_HASH_EMPTY, dtype=torch.int64)
    block_to_hash_slot = torch.full((max_blocks,), PY_HASH_EMPTY, dtype=torch.int32)

    for pool_idx in range(int(coords.shape[0])):
        bx, by, bz = (int(v) for v in coords[pool_idx].tolist())
        entry = pack_hash_entry_host(bx, by, bz, pool_idx)
        slot = spatial_hash_host(bx, by, bz, hash_capacity)
        inserted = False
        for _ in range(hash_capacity):
            if int(hash_table[slot].item()) == PY_HASH_EMPTY:
                hash_table[slot] = entry
                block_to_hash_slot[pool_idx] = slot
                inserted = True
                break
            slot = (slot + 1) % hash_capacity
        if not inserted:
            log_and_raise(
                "Could not rebuild block hash table during import; "
                "hash capacity is full."
            )

    return hash_table, block_to_hash_slot


def prepare_blocks_for_import(
    blocks: Dict[str, torch.Tensor],
    block_metadata: Dict[str, Any],
    *,
    import_weight: Optional[float],
    minimum_tsdf_weight: float,
    block_empty_threshold: float,
) -> Dict[str, torch.Tensor]:
    """Clone and apply the selected confidence policy to block payload tensors."""
    validate_block_payload(blocks, block_metadata)
    prepared = clone_blocks(blocks)
    if import_weight is None:
        validate_recycle_threshold(prepared, block_metadata, block_empty_threshold)
        return prepared

    weight = float(import_weight)
    if weight <= 0.0:
        log_and_raise(f"import_weight must be positive, got {weight}.")
    if weight < float(minimum_tsdf_weight):
        log_and_raise(
            f"import_weight ({weight}) must be >= minimum_tsdf_weight "
            f"({minimum_tsdf_weight})."
        )

    if bool(block_metadata["has_dynamic"]):
        apply_constant_dynamic_weight(prepared, weight)
    if int(block_metadata["feature_dim"]) > 0:
        apply_constant_feature_weight(prepared, weight)

    validate_recycle_threshold(prepared, block_metadata, block_empty_threshold)
    return prepared


def apply_constant_dynamic_weight(
    blocks: Dict[str, torch.Tensor],
    import_weight: float,
) -> None:
    block_data = blocks["block_data"]
    old_weight = block_data[..., 1].float()
    observed = old_weight > 0.0
    if observed.any():
        sdf = torch.zeros_like(old_weight)
        sdf[observed] = block_data[..., 0].float()[observed] / old_weight[observed]
        block_data[..., 0] = (sdf * import_weight).to(dtype=block_data.dtype)
        block_data[..., 1] = observed.to(dtype=block_data.dtype) * import_weight

    block_rgb = blocks["block_rgb"]
    old_rgb_weight = block_rgb[:, 3].float()
    observed_rgb = old_rgb_weight > 0.0
    if observed_rgb.any():
        avg_rgb = torch.zeros_like(block_rgb[:, :3], dtype=torch.float32)
        avg_rgb[observed_rgb] = (
            block_rgb[:, :3].float()[observed_rgb]
            / old_rgb_weight[observed_rgb].unsqueeze(-1)
        )
        block_rgb[:, :3] = (avg_rgb * import_weight).to(dtype=block_rgb.dtype)
        block_rgb[:, 3] = observed_rgb.to(dtype=block_rgb.dtype) * import_weight


def apply_constant_feature_weight(
    blocks: Dict[str, torch.Tensor],
    import_weight: float,
) -> None:
    block_features = blocks["block_features"]
    block_feature_weight = blocks["block_feature_weight"]
    old_weight = block_feature_weight.float()
    observed = old_weight > 0.0
    if not observed.any():
        return
    avg_features = torch.zeros_like(block_features, dtype=torch.float32)
    avg_features[observed] = (
        block_features.float()[observed] / old_weight[observed].unsqueeze(-1)
    )
    block_features[:] = (avg_features * import_weight).to(dtype=block_features.dtype)
    block_feature_weight[:] = observed.to(dtype=block_feature_weight.dtype) * import_weight


def validate_recycle_threshold(
    blocks: Dict[str, torch.Tensor],
    block_metadata: Dict[str, Any],
    block_empty_threshold: float,
) -> None:
    n_blocks = int(blocks["active_block_coords"].shape[0])
    if n_blocks == 0:
        return
    if bool(block_metadata["has_dynamic"]):
        dynamic_sums = blocks["block_data"][..., 1].sum(dim=1, dtype=torch.float32)
    else:
        dynamic_sums = torch.zeros(n_blocks, dtype=torch.float32)
    if bool(block_metadata["has_static"]):
        static_counts = torch.isfinite(blocks["static_block_data"].float()).sum(dim=1)
    else:
        static_counts = torch.zeros(n_blocks, dtype=torch.int64)
    non_empty = (dynamic_sums > 0.0) | (static_counts > 0)
    immediately_recyclable = (
        non_empty & (dynamic_sums < float(block_empty_threshold)) & (static_counts == 0)
    )
    if bool(immediately_recyclable.any()):
        log_and_raise(
            "Imported dynamic blocks would be immediately recyclable. "
            "Increase import_weight or provide higher-confidence source weights."
        )


def require_tensor(
    blocks: Dict[str, torch.Tensor],
    key: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if key not in blocks:
        log_and_raise(f"blocks missing required field {key!r}.")
    value = blocks[key]
    if not isinstance(value, torch.Tensor):
        log_and_raise(
            f"blocks[{key!r}] must be a torch.Tensor, got {type(value).__name__}."
        )
    if value.dtype != dtype:
        log_and_raise(f"blocks[{key!r}] dtype must be {dtype}, got {value.dtype}.")
    return value


def require_shape(tensor: torch.Tensor, key: str, shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != shape:
        log_and_raise(f"{key} must have shape {shape}, got {tuple(tensor.shape)}.")


def require_positive_float(block_metadata: Dict[str, Any], key: str) -> None:
    value = block_metadata.get(key)
    if (
        not isinstance(value, (float, int))
        or isinstance(value, bool)
        or float(value) <= 0.0
    ):
        log_and_raise(f"block_metadata[{key!r}] must be a positive float, got {value!r}.")


def require_positive_int(block_metadata: Dict[str, Any], key: str) -> None:
    value = block_metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        log_and_raise(f"block_metadata[{key!r}] must be a positive int, got {value!r}.")


def require_nonnegative_int(block_metadata: Dict[str, Any], key: str) -> None:
    value = block_metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        log_and_raise(
            f"block_metadata[{key!r}] must be a non-negative int, got {value!r}."
        )


def require_bool(block_metadata: Dict[str, Any], key: str) -> None:
    value = block_metadata.get(key)
    if not isinstance(value, bool):
        log_and_raise(f"block_metadata[{key!r}] must be bool, got {value!r}.")


def require_vec3(block_metadata: Dict[str, Any], key: str) -> None:
    value = block_metadata.get(key)
    if not isinstance(value, list) or len(value) != 3:
        log_and_raise(f"block_metadata[{key!r}] must be a length-3 list.")
    for item in value:
        if not isinstance(item, (float, int)) or isinstance(item, bool):
            log_and_raise(f"block_metadata[{key!r}] must contain numeric values.")


def require_int3(block_metadata: Dict[str, Any], key: str) -> None:
    value = block_metadata.get(key)
    if not isinstance(value, list) or len(value) != 3:
        log_and_raise(f"block_metadata[{key!r}] must be a length-3 list.")
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            log_and_raise(f"block_metadata[{key!r}] must contain positive ints.")


def require_close(source: float, target: float, field_name: str) -> None:
    if not math.isclose(source, target, rel_tol=0.0, abs_tol=1.0e-9):
        log_and_raise(f"{field_name} mismatch: checkpoint={source}, target={target}.")
