# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""cuda.core backend for PBA+ 3D Euclidean Distance Transform.

Launches 5 CUDA kernels (3 unique) to compute an exact 3D Voronoi diagram
using the Parallel Banding Algorithm.  The result is a site_index tensor
where each voxel stores the packed coordinates of its nearest surface site.

Kernel sequence:
    FloodZ  -> MaurerAxis -> ColorAxis -> MaurerAxis -> ColorAxis
    Phase 1    Phase 2a      Phase 2b     Phase 3a      Phase 3b

Phases 2b and 3b perform an X<->Y transpose in the output, so the same
Maurer/Color kernels handle both the Y and X axes.
"""

# Third Party
import torch

# CuRobo
from curobo._src.context import get_runtime
from curobo._src.curobolib.backends.cuda_core_backend.launch_helper import launch_kernel
from curobo._src.curobolib.backends.cuda_core_backend.pba_config import (
    PBAKernelCfg,
    PBALaunchCfg,
)

# Kernel names (fully qualified with C++ namespace)
_NS = "curobo::parallel_banding::"
_FLOOD_Z = _NS + "kernel_flood_z"
_MAURER = _NS + "kernel_maurer_axis"
_COLOR = _NS + "kernel_color_axis"


def _get_compiled_kernels(cache):
    """Compile or retrieve the three PBA+ kernels.

    Returns:
        Tuple of (flood_z, maurer_axis, color_axis) kernel objects.
    """
    cfg = PBAKernelCfg()
    source_files = [cfg.kernel_dir / f for f in cfg.get_kernel_files()]
    include_dirs = cfg.get_include_dirs()
    compile_flags = cfg.get_compile_flags()

    flood_z = cache.get_or_compile_kernel(
        source_files, _FLOOD_Z, include_dirs, compile_flags,
    )
    maurer = cache.get_or_compile_kernel(
        source_files, _MAURER, include_dirs, compile_flags,
    )
    color = cache.get_or_compile_kernel(
        source_files, _COLOR, include_dirs, compile_flags,
    )
    return flood_z, maurer, color


def launch_pba3d(
    site_index: torch.Tensor,
    buffer: torch.Tensor,
    nx: int,
    ny: int,
    nz: int,
    m3: int = 2,
) -> None:
    """Run PBA+ 3D EDT on *site_index* (modified in-place).

    Args:
        site_index: Packed site indices, shape (nx*ny*nz,) or (nx,ny,nz),
            dtype int32.  Sites >= 0, non-sites < 0.
        buffer: Scratch buffer, same size and dtype as site_index.
        nx: Grid size along X (slowest dimension in CuRobo layout).
        ny: Grid size along Y.
        nz: Grid size along Z (fastest dimension in CuRobo layout).
        m3: Color-axis block Y dim (tuning parameter, default 2).
    """
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    flood_z, maurer, color = _get_compiled_kernels(cache)

    pt_stream = torch.cuda.current_stream(site_index.device)
    stream = cache.get_stream_wrapper(pt_stream)

    # PBA axis mapping: sx=nz, sy=ny, sz=nx
    sx, sy, sz = nz, ny, nx

    buf0 = site_index.data_ptr()  # input / scratch A
    buf1 = buffer.data_ptr()      # scratch B

    # Phase 1: Flood Z  (buf0 -> buf1)
    cfg1 = PBALaunchCfg.flood_z(sx, sy)
    launch_kernel(_FLOOD_Z, stream, cfg1, flood_z, buf0, buf1, sx, sy, sz)

    # Phase 2a: Maurer Y  (buf1 -> buf0 as stack)
    cfg2a = PBALaunchCfg.maurer_axis(sx, sz)
    launch_kernel(_MAURER, stream, cfg2a, maurer, buf1, buf0, sx, sy, sz)

    # Phase 2b: Color Y + transpose  (buf0 -> buf1, output dims: sy, sx, sz)
    cfg2b = PBALaunchCfg.color_axis(sx, sz, m3)
    launch_kernel(_COLOR, stream, cfg2b, color, buf0, buf1, sx, sy, sz)

    # Phase 3a: Maurer on transposed data  (buf1 -> buf0)
    #   After transpose: effective dims are (sy, sx, sz)
    cfg3a = PBALaunchCfg.maurer_axis(sy, sz)
    launch_kernel(_MAURER, stream, cfg3a, maurer, buf1, buf0, sy, sx, sz)

    # Phase 3b: Color + transpose back  (buf0 -> buf1, restores original layout)
    cfg3b = PBALaunchCfg.color_axis(sy, sz, m3)
    launch_kernel(_COLOR, stream, cfg3b, color, buf0, buf1, sy, sx, sz)

    # Result is in buffer; copy back to site_index
    site_index.view(-1).copy_(buffer.view(-1))
