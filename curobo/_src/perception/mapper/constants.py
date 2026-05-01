# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Shared constants for block-sparse TSDF / ESDF mapping.

This module is the single source of truth for values that appear in
more than one mapper submodule:

- Hash-table bit layout: key/value widths, offsets, masks, primes,
  sentinel values. Exported both as plain Python ints (for host-side
  code and ``torch`` operations) and as ``wp.constant`` values (for
  device-side Warp kernels).
- ``PENDING_POOL_IDX`` / ``MAX_POOL_IDX``: pool-index sentinel and the
  derived ceiling.
- ``REFERENCE_BLOCK_SIZE``: historical anchor (8) used by allocation
  heuristics and recycle thresholds to keep behavior invariant across
  configurable ``block_size`` values.

Choosing a ``block_size``:
    ``block_size`` is now a per-instance configuration value on
    :class:`~curobo._src.perception.mapper.mapper_cfg.MapperCfg` /
    :class:`~curobo._src.perception.mapper.storage.BlockSparseTSDFCfg`.
    Set it at construction time; Warp kernels specialize on the value
    automatically:

    .. code-block:: python

        from curobo._src.perception.mapper.mapper import Mapper
        from curobo._src.perception.mapper.mapper_cfg import MapperCfg
        mapper = Mapper(MapperCfg(..., block_size=4))

    Two mappers with different ``block_size`` values can coexist in
    the same process. The Python kernel bundle is rebuilt on demand,
    while Warp owns compiled-kernel reuse. See
    :mod:`curobo._src.perception.mapper.kernel.builder` for details.
"""

from dataclasses import dataclass
from typing import Tuple

import warp as wp

from curobo._src.util.logging import log_and_raise

# =============================================================================
# Block Size Validation
# =============================================================================

#: Historical reference value (``8``) used by block-size-invariant
#: scaling in :mod:`block_allocation` (voxel-budget floor, default
#: pool target) and :mod:`kernel.wp_decay` (recycle threshold). Kept
#: as a fixed anchor rather than following the active ``block_size``
#: so scaling relative to the v0.7 default is stable.
REFERENCE_BLOCK_SIZE: int = 8

# Maximum permitted block_size.
# Rationale (picking the tightest remaining constraints):
#   1. Per-block memory: block_data is (max_blocks, block_size**3, 2)
#      fp16. At block_size=32 the fp16 TSDF channel alone needs
#      ~12.5 GB for 100K blocks; block_size=64 would need ~100 GB.
#   2. CUDA block_dim: wp.launch uses block_dim=block_size (not
#      cubed), so the 1024 threads/CUDA-block hardware cap only bites
#      at block_size > 1024 - not the binding constraint.
#   3. Kernel specialization and downstream tooling assume the mapper's
#      supported block-size set is bounded to the single-voxel case plus
#      the historical powers of two.
_BLOCK_SIZE_MAX: int = 32


def _validate_block_size(block_size: int) -> None:
    """Validate that a ``block_size`` value satisfies kernel, launch, and hardware limits.

    Called by the block-sparse kernel builder before any Warp kernel
    is compiled. Fails fast with a descriptive error routed
    through :func:`curobo._src.util.logging.log_and_raise`.

    Constraints enforced:
        - int (not bool / float / numpy scalar).
        - >= 1.
        - 1 or a power of 2 (preserves the historical layout, and keeps
          div/mod by ``block_size`` lowerable to shift/mask in Warp
          codegen for the non-degenerate block sizes).
        - <= ``_BLOCK_SIZE_MAX`` (see comment above for the combined
          launch-dim + memory rationale).
    """
    if not isinstance(block_size, int) or isinstance(block_size, bool):
        log_and_raise(
            f"block_size must be a plain int, got {type(block_size).__name__} ({block_size!r})."
        )
    if block_size < 1:
        log_and_raise(f"block_size must be >= 1, got {block_size}.")
    if block_size & (block_size - 1) != 0:
        log_and_raise(
            f"block_size must be 1 or a power of 2 (2, 4, 8, 16, 32), got "
            f"{block_size}. Non-power-of-2 values defeat shift/mask "
            f"lowering of div/mod in kernel codegen and may break "
            f"downstream tooling that assumes the historical layout."
        )
    if block_size > _BLOCK_SIZE_MAX:
        log_and_raise(
            f"block_size must be <= {_BLOCK_SIZE_MAX}, got {block_size}. "
            f"Larger values push per-block memory to impractical sizes "
            f"(>= 32 KB/block per channel) and are outside the mapper's "
            f"supported kernel-specialization set."
        )


def _validate_feature_channels_per_thread(feature_channels_per_thread: int) -> None:
    """Validate feature-channel grouping for the generated integration kernel.

    The grouped feature integration kernel specializes on this value, so it
    must be a positive plain int shared by Python launch grouping and Warp
    thread decoding.
    """
    if not isinstance(feature_channels_per_thread, int) or isinstance(
        feature_channels_per_thread, bool
    ):
        log_and_raise(
            "feature_channels_per_thread must be a plain int, got "
            f"{type(feature_channels_per_thread).__name__} "
            f"({feature_channels_per_thread!r})."
        )
    if feature_channels_per_thread < 1:
        log_and_raise(
            "feature_channels_per_thread must be >= 1, got "
            f"{feature_channels_per_thread}."
        )


def _validate_feature_grid_shape(
    feature_dim: int,
    feature_grid_height: int | None,
    feature_grid_width: int | None,
) -> None:
    """Validate the compile-time feature-grid shape contract.

    The feature kernels specialize on feature-grid height/width. When
    features are enabled, callers must provide both dimensions explicitly;
    when features are disabled, both dimensions must be omitted.
    """
    if feature_dim < 0:
        log_and_raise(f"feature_dim must be >= 0, got {feature_dim}.")

    has_height = feature_grid_height is not None
    has_width = feature_grid_width is not None
    if has_height != has_width:
        log_and_raise(
            "feature_grid_height and feature_grid_width must be specified together."
        )
    if feature_dim == 0:
        if has_height:
            log_and_raise(
                "feature_grid_height/feature_grid_width require feature_dim > 0."
            )
        return

    if not has_height:
        log_and_raise(
            "feature_dim > 0 requires feature_grid_height and feature_grid_width."
        )
    if not isinstance(feature_grid_height, int) or isinstance(feature_grid_height, bool):
        log_and_raise(
            "feature_grid_height must be a plain int, got "
            f"{type(feature_grid_height).__name__} ({feature_grid_height!r})."
        )
    if not isinstance(feature_grid_width, int) or isinstance(feature_grid_width, bool):
        log_and_raise(
            "feature_grid_width must be a plain int, got "
            f"{type(feature_grid_width).__name__} ({feature_grid_width!r})."
        )
    if feature_grid_height <= 0 or feature_grid_width <= 0:
        log_and_raise(
            "feature_grid_height and feature_grid_width must be positive, got "
            f"{feature_grid_height}x{feature_grid_width}."
        )


FEATURE_INTEGRATION_KERNEL_MODES: tuple[str, str, str] = ("auto", "grouped", "tiled")


def _validate_feature_integration_kernel(feature_integration_kernel: str) -> None:
    """Validate the public feature-integration launch policy."""
    if not isinstance(feature_integration_kernel, str):
        log_and_raise(
            "feature_integration_kernel must be one of "
            f"{FEATURE_INTEGRATION_KERNEL_MODES}, got "
            f"{type(feature_integration_kernel).__name__}."
        )
    if feature_integration_kernel not in FEATURE_INTEGRATION_KERNEL_MODES:
        log_and_raise(
            "feature_integration_kernel must be one of "
            f"{FEATURE_INTEGRATION_KERNEL_MODES}, got "
            f"{feature_integration_kernel!r}."
        )


def resolve_feature_integration_kernel(
    feature_integration_kernel: str,
    feature_dim: int,
    support_capacity: int,
) -> bool:
    """Resolve public feature integration policy to the low-level tiled bool."""
    _validate_feature_integration_kernel(feature_integration_kernel)
    if feature_integration_kernel == "tiled":
        return True
    if feature_integration_kernel == "grouped":
        return False

    if feature_dim >= 512:
        return True
    if feature_dim >= 128 and support_capacity <= 8:
        return True
    if feature_dim >= 64 and support_capacity <= 4:
        return True
    return False


# =============================================================================
# Hash Table Bit Layout (64-bit packed entries)
# =============================================================================
#
# Plain Python ints are the source of truth; ``wp.constant`` values are
# derived from them so host and device code cannot drift.


@dataclass(frozen=True)
class HashLayout:
    """Immutable packed hash-entry bit layout.

    The mapper has one supported runtime layout. This value object keeps
    masks, shifts, sentinels, and coordinate ranges derived from the
    construction-time split so host validation and Warp kernel factories
    use the same contract.
    """

    coord_bits_xyz: Tuple[int, int, int]

    def __post_init__(self) -> None:
        bits = self.coord_bits_xyz
        if len(bits) != 3:
            raise ValueError(f"coord_bits_xyz must contain 3 entries, got {bits!r}.")
        for axis, bit_count in zip(("x", "y", "z"), bits):
            if not isinstance(bit_count, int) or isinstance(bit_count, bool):
                raise ValueError(f"{axis} coordinate bit count must be an int, got {bit_count!r}.")
            if bit_count <= 0:
                raise ValueError(f"{axis} coordinate bit count must be positive, got {bit_count}.")
        if sum(bits) >= 64:
            raise ValueError(
                f"HashLayout must leave at least one value bit; got coord_bits_xyz={bits!r}."
            )
        if self.pending_pool_idx > 2_147_483_647:
            raise ValueError(
                "HashLayout pending_pool_idx must fit int32 for Warp kernels, got "
                f"{self.pending_pool_idx}."
            )

    @property
    def value_bits(self) -> int:
        return 64 - sum(self.coord_bits_xyz)

    @property
    def name(self) -> str:
        x_bits, y_bits, z_bits = self.coord_bits_xyz
        return f"x{x_bits}y{y_bits}z{z_bits}v{self.value_bits}"

    @property
    def coord_masks_xyz(self) -> Tuple[int, int, int]:
        return tuple((1 << bits) - 1 for bits in self.coord_bits_xyz)

    @property
    def coord_bias_xyz(self) -> Tuple[int, int, int]:
        return tuple(1 << (bits - 1) for bits in self.coord_bits_xyz)

    @property
    def coord_min_xyz(self) -> Tuple[int, int, int]:
        return tuple(-bias for bias in self.coord_bias_xyz)

    @property
    def coord_max_xyz(self) -> Tuple[int, int, int]:
        return tuple(bias - 1 for bias in self.coord_bias_xyz)

    @property
    def value_mask(self) -> int:
        return (1 << self.value_bits) - 1

    @property
    def key_mask(self) -> int:
        return ((1 << 64) - 1) ^ self.value_mask

    @property
    def key_mask_signed(self) -> int:
        return -1 << self.value_bits

    @property
    def z_shift(self) -> int:
        return self.value_bits

    @property
    def y_shift(self) -> int:
        return self.z_shift + self.coord_bits_xyz[2]

    @property
    def x_shift(self) -> int:
        return self.y_shift + self.coord_bits_xyz[1]

    @property
    def pending_pool_idx(self) -> int:
        return self.value_mask

    @property
    def max_pool_idx(self) -> int:
        return self.pending_pool_idx - 1


DEFAULT_HASH_LAYOUT = HashLayout(coord_bits_xyz=(13, 13, 13))


def _ceil_div_positive(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def validate_grid_shape_for_hash_layout(
    grid_shape: Tuple[int, int, int] | None,
    block_size: int,
    *,
    layout: HashLayout = DEFAULT_HASH_LAYOUT,
    field_name: str = "grid_shape",
) -> Tuple[int, int, int]:
    """Validate and normalize a bounded TSDF grid shape.

    ``grid_shape`` is stored as ``(nz, ny, nx)``. The packed layout is
    ordered ``x, y, z``, so validation compares ``nx`` to the X field,
    ``ny`` to Y, and ``nz`` to Z.
    """
    if grid_shape is None:
        log_and_raise(f"{field_name} is required for bounded block-sparse mapping.")
    if len(grid_shape) != 3:
        log_and_raise(f"{field_name} must be a 3-tuple (nz, ny, nx), got {grid_shape!r}.")

    try:
        nz, ny, nx = (int(v) for v in grid_shape)
    except (TypeError, ValueError):
        log_and_raise(f"{field_name} must contain integer dimensions, got {grid_shape!r}.")

    if nz <= 0 or ny <= 0 or nx <= 0:
        log_and_raise(f"{field_name} dimensions must be positive, got {(nz, ny, nx)}.")

    blocks_x = _ceil_div_positive(nx, block_size)
    blocks_y = _ceil_div_positive(ny, block_size)
    blocks_z = _ceil_div_positive(nz, block_size)
    block_counts = (blocks_x, blocks_y, blocks_z)
    max_blocks_per_axis = tuple(1 << bits for bits in layout.coord_bits_xyz)
    for axis, count, limit in zip(("x", "y", "z"), block_counts, max_blocks_per_axis):
        if count > limit:
            log_and_raise(
                f"{field_name} requires {count} blocks along {axis}, exceeding "
                f"the {layout.name} hash layout limit of {limit} blocks/axis. "
                f"grid_shape={(nz, ny, nx)}, block_size={block_size}."
            )

    return (nz, ny, nx)


# --- Entry sentinels --------------------------------------------------------

#: Empty hash slot sentinel (all 1s as signed int64 = -1).
PY_HASH_EMPTY: int = -1

#: Deleted-entry sentinel.
PY_HASH_TOMBSTONE: int = -2

HASH_EMPTY = wp.constant(wp.int64(PY_HASH_EMPTY))
HASH_TOMBSTONE = wp.constant(wp.int64(PY_HASH_TOMBSTONE))

# --- Spatial hash primes (Teschner et al.) ----------------------------------

PY_HASH_PRIME_X: int = 73856093
PY_HASH_PRIME_Y: int = 19349663
PY_HASH_PRIME_Z: int = 83492791

HASH_PRIME_X = wp.constant(wp.int64(PY_HASH_PRIME_X))
HASH_PRIME_Y = wp.constant(wp.int64(PY_HASH_PRIME_Y))
HASH_PRIME_Z = wp.constant(wp.int64(PY_HASH_PRIME_Z))

# --- Block-coordinate encoding (13-bit signed per axis) ---------------------

#: Bits per axis in the packed key.
PY_COORD_BITS: int = DEFAULT_HASH_LAYOUT.coord_bits_xyz[0]

#: Offset added to signed coords to encode them as unsigned ints
#: (gives a signed range of [-4096, 4095]).
PY_COORD_OFFSET: int = DEFAULT_HASH_LAYOUT.coord_bias_xyz[0]

#: Mask for one coordinate axis (13 bits).
PY_COORD_MASK: int = DEFAULT_HASH_LAYOUT.coord_masks_xyz[0]

COORD_BITS = wp.constant(PY_COORD_BITS)
COORD_OFFSET = wp.constant(wp.int64(PY_COORD_OFFSET))
COORD_MASK = wp.constant(wp.int64(PY_COORD_MASK))

# Back-compat aliases used by existing kernels.
BLOCK_KEY_BITS = COORD_BITS
BLOCK_KEY_OFFSET = COORD_OFFSET
BLOCK_KEY_MASK = COORD_MASK

# --- Pool-index field (lower 25 bits of the packed entry) -------------------

#: Bits reserved for the pool index.
PY_VALUE_BITS: int = DEFAULT_HASH_LAYOUT.value_bits

#: Mask covering the 25-bit pool_idx range.
PY_VALUE_MASK: int = DEFAULT_HASH_LAYOUT.value_mask

VALUE_BITS = wp.constant(PY_VALUE_BITS)
VALUE_MASK = wp.constant(wp.int64(PY_VALUE_MASK))

# --- Bit-shift positions ----------------------------------------------------

#: Shift for the Z coordinate in the packed entry.
PY_Z_SHIFT: int = DEFAULT_HASH_LAYOUT.z_shift
PY_Y_SHIFT: int = DEFAULT_HASH_LAYOUT.y_shift
PY_X_SHIFT: int = DEFAULT_HASH_LAYOUT.x_shift

Z_SHIFT = wp.constant(wp.int64(PY_Z_SHIFT))
Y_SHIFT = wp.constant(wp.int64(PY_Y_SHIFT))
X_SHIFT = wp.constant(wp.int64(PY_X_SHIFT))

#: Upper-39-bits mask (covers the three 13-bit coordinate fields).
PY_KEY_MASK: int = DEFAULT_HASH_LAYOUT.key_mask

KEY_MASK = wp.constant(wp.int64(DEFAULT_HASH_LAYOUT.key_mask_signed))

# --- Utility mask -----------------------------------------------------------

#: Used to ensure positive hash results across signed/unsigned conversions.
PY_POSITIVE_MASK: int = 0x7FFFFFFFFFFFFFFF


# =============================================================================
# Pool-Index Sentinels
# =============================================================================

#: Marker stored in ``hash_table`` while an entry is being allocated. Same
#: value as the active layout's all-ones value mask, so it cannot be
#: produced by a real ``pool_idx`` (they are strictly below this value).
PENDING_POOL_IDX: int = DEFAULT_HASH_LAYOUT.pending_pool_idx  # 0x1FFFFFF = 33_554_431

#: ``wp.int32`` form of :data:`PENDING_POOL_IDX` for use inside Warp
#: kernels (``wp.constant(wp.int32(...))`` at module scope).
PENDING_POOL_IDX_WP = wp.constant(wp.int32(PENDING_POOL_IDX))

#: Maximum representable real pool index. Any ``max_blocks`` >=
#: ``PENDING_POOL_IDX`` would alias a real block onto the sentinel.
MAX_POOL_IDX: int = DEFAULT_HASH_LAYOUT.max_pool_idx  # 33_554_430
