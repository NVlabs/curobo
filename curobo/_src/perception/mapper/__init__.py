# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Mapper module for volumetric mapping.

This module provides tools for integrating depth images into TSDF/ESDF
representations using block-sparse storage.

Main classes:
    High-Level API:
        - Mapper: Unified mapper facade with block-sparse storage.
        - MapperCfg: Configuration for Mapper with intuitive physical parameters.

    Block-Sparse (Primary):
        - BlockSparseTSDF: Block-sparse TSDF storage.
        - BlockSparseTSDFIntegrator: TSDF-only integrator.
        - BlockSparseESDFIntegrator: Fused TSDF+ESDF pipeline.
        - BlockSparseTSDFRenderer: Render depth/normal images via raycasting.
        - BlockSparseRaycastPoseRefiner: ICP using TSDF raycasting.

    Shared (EDT solvers):
        - JumpFloodingEDT: Approximate EDT via Jump Flooding Algorithm.
        - ParallelBandingEDT: Exact EDT via Parallel Banding Algorithm.
"""

# High-level API
# EDT solvers
from curobo._src.perception.mapper.esdf.edt_jump_flooding import JumpFloodingEDT
from curobo._src.perception.mapper.esdf.edt_parallel_banding import ParallelBandingEDT
from curobo._src.perception.mapper.integrator_esdf import (
    BlockSparseESDFIntegrator,
    BlockSparseESDFIntegratorCfg,
)

# Block-sparse integrators
from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.perception.mapper.mapper import Mapper
from curobo._src.perception.mapper.mapper_cfg import MapperCfg

# Block-sparse mesh extraction
from curobo._src.perception.mapper.mesh_extractor import (
    extract_mesh_block_sparse,
)

# Block-sparse pose refinement
from curobo._src.perception.mapper.pose_refiner import (
    BlockSparseRaycastPoseRefiner,
    BlockSparseRaycastRefinerCfg,
)

# Block-sparse renderer
# Renderer utilities
from curobo._src.perception.mapper.renderer import (
    BlockSparseTSDFRenderer,
    depth_to_colormap,
    normals_to_colormap,
)

# Block-sparse storage
from curobo._src.perception.mapper.storage import (
    BlockSparseTSDF,
    BlockSparseTSDFCfg,
)

__all__ = [
    # High-level API
    "Mapper",
    "MapperCfg",
    # Block-sparse storage
    "BlockSparseTSDF",
    "BlockSparseTSDFCfg",
    # Block-sparse integrators
    "BlockSparseTSDFIntegrator",
    "BlockSparseTSDFIntegratorCfg",
    "BlockSparseESDFIntegrator",
    "BlockSparseESDFIntegratorCfg",
    # Block-sparse renderer
    "BlockSparseTSDFRenderer",
    # Block-sparse pose refinement
    "BlockSparseRaycastPoseRefiner",
    "BlockSparseRaycastRefinerCfg",
    # Block-sparse mesh extraction
    "extract_mesh_block_sparse",
    # Renderer utilities
    "depth_to_colormap",
    "normals_to_colormap",
    # EDT solvers
    "JumpFloodingEDT",
    "ParallelBandingEDT",
]

