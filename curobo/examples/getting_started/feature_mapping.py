# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Fuse neural image features into a TSDF map and query semantic regions.

This tutorial extends :mod:`curobo.examples.getting_started.volumetric_mapping`
with a learned feature channel. cuRobo still integrates depth frames into a
block-sparse Truncated Signed Distance Field (TSDF), but each RGB frame is also
encoded by NVIDIA C-RADIO and passed to :class:`~curobo.types.CameraObservation`
as ``feature_grid``. The mapper fuses those patch features into the allocated
TSDF blocks, so later queries can find parts of the 3D map that are visually or
semantically similar.

.. raw:: html

   <figure style="margin:0 0 1.5em;">
     <video autoplay loop muted playsinline style="width:100%;border-radius:6px;">
       <source src="../videos/feature_mapping_integration.webm" type="video/webm">
     </video>
     <figcaption style="text-align:center;font-style:italic;margin-top:0.4em;">Feature Integration</figcaption>
   </figure>

C-RADIO (Reduce All Domains Into One) distills multiple vision foundation
models, including DINOv2, SAM, CLIP, and SigLIP, into one backbone. This example
uses the C-RADIO v3-B checkpoint and its per-image patch embeddings in two
beginner-friendly ways:

- Project image or map features to RGB with Principal Component Analysis (PCA)
  so feature clusters can be inspected visually.
- When the viewer is enabled, project block features through the fixed SigLIP
  adaptor and match them against text prompts such as ``table`` or ``chair``.

This example downloads C-RADIO v3-B through ``torch.hub`` on first use. The
first run must be able to reach ``NVlabs/RADIO`` on GitHub, download the
checkpoint, and install any missing RADIO dependencies.

By the end of this tutorial you will have:

- Loaded an RGB-D sequence from Sun3D
- Extracted C-RADIO v3-B patch features for each selected RGB frame
- Fused depth, color, and learned features into a TSDF map
- Saved side-by-side ``RGB | PCA(features)`` images for quick inspection
- Visualized the map and highlighted blocks that match a text prompt when
  ``--visualize`` is enabled

Before starting
------------------

Install the extra feature mapping dependencies. If your environment needs
Hugging Face authentication for checkpoint downloads, export ``HF_TOKEN`` before
running the example:

.. code-block:: bash

   export HF_TOKEN=<your_huggingface_token>
   uv pip install timm transformers torchvision einops

How the mapper uses features
-------------------------------

The feature mapper follows the same geometry path as the volumetric mapping
tutorial, with one extra input: a lower-resolution grid of learned patch
features from the RGB image.

.. graphviz::
   :caption: RGB-D feature mapping data flow

   digraph FeatureMapping {
      rankdir=LR;
      edge [color="#2B4162", fontsize=10];
      node [shape="box", style="rounded, filled", fontsize=12, color="#cccccc"];

      rgb [label="RGB image", color="#708090", fontcolor="white"];
      depth [label="Depth image", color="#708090", fontcolor="white"];
      camera [label="Camera pose\\n+ intrinsics", color="#708090", fontcolor="white"];
      radio [label="C-RADIO\\npatch features", color="#558c8c", fontcolor="white"];
      obs [label="CameraObservation\\ndepth + RGB + feature_grid",
           color="#76b900", fontcolor="white"];
      mapper [label="Mapper.integrate()\\nblock-sparse TSDF",
              color="#76b900", fontcolor="white"];
      blocks [label="TSDF blocks\\ngeometry + color + features",
              color="#558c8c", fontcolor="white"];
      pca [label="PCA colors\\nfeature clusters", color="#708090", fontcolor="white"];
      text [label="Text matching\\nSigLIP adaptor", color="#708090", fontcolor="white"];

      rgb -> radio -> obs;
      depth -> obs;
      camera -> obs;
      obs -> mapper -> blocks;
      blocks -> pca;
      blocks -> text;
   }

Step 1: Download the dataset
-------------------------------

This tutorial uses the same `Sun3D <http://sun3d.cs.princeton.edu/>`_ indoor
RGB-D scene as the volumetric mapping tutorial. It contains color images, depth
maps, camera intrinsics, and ground-truth camera poses.

Quick start (downloads one scene, about 1400 MB):

.. code-block:: bash

   wget http://3dvision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/sun3d-mit_76_studyroom-76-1studyroom2.zip
   mkdir -p datasets/sun3d
   unzip sun3d-mit_76_studyroom-76-1studyroom2.zip -d datasets/sun3d

The extracted directory should look like::

    datasets/sun3d/sun3d-mit_76_studyroom-76-1studyroom2/
        camera-intrinsics.txt
        <sequence_name>/
            000001.color.png
            000001.depth.png
            000001.pose.txt
            ...

Step 2: Run a quick feature-fusion pass
------------------------------------------

Start with a small number of frames because C-RADIO inference is heavier than
plain depth integration:

.. code-block:: bash

   python -m curobo.examples.getting_started.feature_mapping \\
       --root ./datasets/sun3d/sun3d-mit_76_studyroom-76-1studyroom2 \\
       --num-frames 50 \\
       --stride 10 \\
       --save-pca

When ``--save-pca`` is enabled, the tutorial writes side-by-side RGB and
feature-PCA panels to ``~/.cache/curobo/examples/feature_mapping/``. The colors
are not object labels; they are a three-dimensional PCA projection of
high-dimensional feature vectors, so nearby colors usually indicate similar
visual embeddings.

Step 3: Inspect the map interactively
----------------------------------------

Add ``--visualize`` to open a `Viser <https://viser.studio>`_ server at
http://localhost:8080:

.. code-block:: bash

   python -m curobo.examples.getting_started.feature_mapping \\
       --root ./datasets/sun3d/sun3d-mit_76_studyroom-76-1studyroom2 \\
       --num-frames 100 \\
       --stride 5 \\
       --visualize

The viewer shows:

- ``/reconstruction/features_pca``: occupied voxels colored by fused C-RADIO
  features projected through a map-wide PCA basis.
- ``/reconstruction/rgb``: occupied voxels colored by the TSDF color channel.
  This layer is hidden by default and can be toggled from the scene tree.
- ``Current RGB`` and ``Current Feature PCA`` panels for the latest frame.


.. raw:: html

   <figure style="margin:0 0 1.5em;">
     <video autoplay loop muted playsinline style="width:100%;border-radius:6px;">
       <source src="../videos/feature_mapping_integration.webm" type="video/webm">
     </video>
     <figcaption style="text-align:center;font-style:italic;margin-top:0.4em;">Feature Integration</figcaption>
   </figure>


Step 4: Try text matching
----------------------------

Use ``--visualize`` to open the Text Matching panel. The example uses the
C-RADIO v3-B SigLIP adaptor for text queries:

.. code-block:: bash

   python -m curobo.examples.getting_started.feature_mapping \\
       --root ./datasets/sun3d/sun3d-mit_76_studyroom-76-1studyroom2 \\
       --num-frames 100 \\
       --stride 5 \\
       --visualize

Enter a prompt in the panel to highlight the top matching TSDF blocks under
``/reconstruction/text_matched``. ``Clear Matches`` demonstrates how matched
blocks can be removed from the dynamic map. For a geometric clearing example,
``--clear-aabb xmin ymin zmin xmax ymax zmax`` clears all allocated blocks that
intersect the given world-space bounds in meters.

.. raw:: html

   <figure style="margin:0 0 1.5em;">
     <video autoplay loop muted playsinline controls style="width:100%;border-radius:6px;">
       <source src="../videos/feature_mapping_text_align.webm" type="video/webm">
     </video>
     <figcaption style="text-align:center;font-style:italic;margin-top:0.4em;">Text Feature Alignment</figcaption>
   </figure>

Step 5: Check the output
---------------------------

When the tutorial finishes successfully you will see output similar to::

    Loading Sun3D from ./datasets/sun3d...
    Found 200 frames
    Loading C-RADIO (c-radio_v3-B) via NVlabs/RADIO torch.hub...
    Feature dim: 768
    Mapper initialized: 64.0 MB
    integrating: 100%|...

    Mapper memory: 64.0 MB
    PCA panels saved to: ~/.cache/curobo/examples/feature_mapping
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from curobo import runtime
from curobo.examples.getting_started.volumetric_mapping import Sun3dDataset
from curobo._src.util.logging import log_warn
from curobo.perception import (
    FilterDepth,
    Mapper,
    MapperCfg,
)
from curobo.profiling import CudaEventTimer
from curobo.types import CameraObservation, Pose


SUN3D_IMAGE_SHAPE = (480, 640)
MAP_EXTENT_METERS_XYZ = (11.0, 7.0, 5.0)
MAX_RECONSTRUCTION_POINTS = 200_000
MAX_TEXT_MATCH_POINTS = 500_000
RADIO_MODEL_NAME = "c-radio_v3-B"
TEXT_ADAPTOR_NAME = "siglip2"
TEXT_TOP_K = 500
TEXT_MIN_SCORE = 0.05


class CRadioInference:
    """Own all C-RADIO neural-network inference used by this tutorial.

    The mapper itself is not a neural network: it fuses depth, color, and feature
    tensors into a TSDF map. This class keeps the learned pieces together so the
    rest of the example can treat them as three simple operations:

    1. extract patch features from an RGB image;
    2. optionally encode text with the requested RADIO adaptor;
    3. optionally project map features into the adaptor's text-aligned space.
    """

    def __init__(
        self,
        model_name: str = RADIO_MODEL_NAME,
        device: str = "cuda:0",
        text_adaptor_name: "str | None" = None,
    ):
        self.device = device
        self.model_name = model_name
        self.text_adaptor_name = text_adaptor_name
        adaptor_names = [self.text_adaptor_name] if self.text_adaptor_name else None
        print(f"Adaptor names: {adaptor_names}")

        hub_version = self.resolve_torchhub_version(model_name)
        self.model = (
            torch.hub.load(
                "NVlabs/RADIO",
                "radio_model",
                source="github",
                version=hub_version,
                progress=True,
                skip_validation=True,
                adaptor_names=adaptor_names,
            )
            .eval()
            .to(device)
        )
        self.patch_size = int(getattr(self.model, "patch_size", 16))

        self.text_adaptor = None
        self.tokenizer = None
        self._encode_text = None
        if self.text_adaptor_name is not None:
            self.text_adaptor = self._resolve_text_adaptor(self.text_adaptor_name)
            self.tokenizer = getattr(self.text_adaptor, "tokenizer", None)
            self._encode_text = getattr(self.text_adaptor, "encode_text", None)
            if self.tokenizer is None or not callable(self._encode_text):
                raise RuntimeError(
                    f"RADIO adaptor '{self.text_adaptor_name}' must expose both "
                    "tokenizer and encode_text for torchhub-only text matching. "
                    "Pick a hardcoded adaptor with a text tower."
                )

    @staticmethod
    def resolve_torchhub_version(model_name: str) -> str:
        """Map a C-RADIO model id to NVlabs/RADIO's torch.hub version key."""
        version = model_name.strip()
        if "/" in version:
            _, version = version.split("/", 1)
        version = version.lower()
        if version.startswith("c-radio_v"):
            return version
        if version.startswith("c-radiov"):
            return version.replace("c-radiov", "c-radio_v", 1)
        raise ValueError(
            f"Could not map model '{model_name}' to a RADIO torch.hub version. "
            "Use a RADIO hub version like 'c-radio_v3-B'."
        )

    @torch.inference_mode()
    def extract_patch_features(self, rgb_uint8: torch.Tensor) -> torch.Tensor:
        """Extract patch features from one RGB image.

        Args:
            rgb_uint8: ``(H, W, 3)`` uint8 image on ``self.device``.

        Returns:
            ``(H_p, W_p, D)`` float32 feature tensor, where
            ``H_p = target_h // patch_size`` and similarly for ``W_p``.
        """
        H, W = rgb_uint8.shape[:2]
        target_h, target_w = self.model.get_nearest_supported_resolution(H, W)

        img = rgb_uint8.permute(2, 0, 1).float() / 255.0
        img = F.interpolate(
            img.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        output = self.model(img)
        if isinstance(output, dict):
            if "backbone" not in output:
                raise RuntimeError(
                    f"Expected RADIO output to contain 'backbone', got keys "
                    f"{list(output.keys())}."
                )
            output = output["backbone"]
        # RADIO returns (summary, features); newer versions use a RadioOutput
        # named tuple with a ``features`` attribute. Handle both formats.
        features = getattr(output, "features", None)
        if features is None:
            _, features = output  # (summary, features)

        ps = self.patch_size
        H_p = target_h // ps
        W_p = target_w // ps
        return features[0].view(H_p, W_p, -1).contiguous()

    def _resolve_text_adaptor(self, adaptor_name: str):
        """Locate a RADIO teacher-adaptor submodule across runtime variants."""
        for attr in ("adaptors", "adapters", "_adaptors"):
            registry = getattr(self.model, attr, None)
            if registry is not None and adaptor_name in registry:
                return registry[adaptor_name]
        available = {}
        for attr in ("adaptors", "adapters", "_adaptors"):
            registry = getattr(self.model, attr, None)
            if registry is not None:
                available[attr] = list(registry.keys())
        raise RuntimeError(
            f"Could not find adaptor '{adaptor_name}' on RADIO model. "
            f"Available registries: {available or 'none found'}. "
            "Inspect with dir(feature_model.model) and update TEXT_ADAPTOR_NAME."
        )

    def _project_through_text_adaptor(self, features: torch.Tensor) -> torch.Tensor:
        """Run the loaded RADIO text adaptor on ``(N, D_radio)`` block features."""
        if self.text_adaptor is None:
            raise RuntimeError("Text matching was requested, but no text adaptor is loaded.")

        # RADIO adaptor APIs vary across checkpoint variants. Try the common
        # projection entry points before falling back to calling the adaptor.
        for attr in ("head_mlp", "feat_mlp", "head"):
            sub = getattr(self.text_adaptor, attr, None)
            if sub is not None and callable(sub):
                return sub(features)
        if callable(self.text_adaptor):
            try:
                out = self.text_adaptor(features)
            except TypeError:
                # Some adaptors want (summary, features); synthesize a summary.
                summary = features.mean(dim=0, keepdim=True)
                out = self.text_adaptor(summary, features.unsqueeze(0))
                if isinstance(out, tuple):
                    out = out[1]
                    if out.dim() == 3:
                        out = out[0]
            if isinstance(out, tuple):
                out = out[1] if len(out) > 1 else out[0]
            return out
        raise RuntimeError(
            f"Adaptor of type {type(self.text_adaptor).__name__} has no known "
            "entry point (head_mlp / feat_mlp / head / __call__)."
        )

    @torch.inference_mode()
    def encode_text(self, text) -> torch.Tensor:
        """Encode one or more strings to ``(N, D_teacher)`` L2-normalized features."""
        if self.tokenizer is None or self._encode_text is None:
            raise RuntimeError(
                "Text matching was requested, but the loaded adaptor has no text tower."
            )
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenizer(text)
        if hasattr(tokens, "to"):
            tokens = tokens.to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            try:
                feats = self._encode_text(tokens, normalize=True)
            except TypeError:
                feats = self._encode_text(tokens)
                feats = F.normalize(feats, dim=-1)
        return feats

    @torch.inference_mode()
    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        """Project ``(N, D_radio)`` map features to L2-normalized teacher features."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self._project_through_text_adaptor(features)
            out = F.normalize(out, dim=-1)
        return out


def _pca_basis(
    centered: torch.Tensor,
    prev_basis: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Orthonormal ``(D, 3)`` basis of the top-3 principal directions.

    Built from a single ``torch.linalg.eigh`` on the ``(D, D)``
    covariance matrix. Any positive-semidefinite matrix has an
    orthonormal eigenbasis, so this path also works for singular /
    rank-deficient inputs (``N < D``, duplicate rows, all-zero
    features): the unused directions land in the zero eigenspace and
    contribute nothing to the projection, exactly what we want for a
    visualization-only PCA.

    The returned basis is always ``(D, 3)`` and orthonormal. When
    fewer than three real PCs exist (e.g. ``D < 3``), trailing columns
    are canonical axes so downstream projection stays well-defined.
    If ``prev_basis`` is compatible, columns are sign-aligned to it so
    per-frame re-fits do not swap red/cyan as new data extends the
    span. PCA is only defined up to a sign per axis.
    """
    N, D = centered.shape
    device, dtype = centered.device, centered.dtype

    basis = torch.zeros((D, 3), device=device, dtype=dtype)
    for i in range(min(D, 3)):
        basis[i, i] = 1.0

    if D > 0 and N > 0:
        try:
            cov = centered.T @ centered
            if N > 1:
                cov = cov / (N - 1)
            cov = 0.5 * (cov + cov.T)  # symmetrize to kill fp drift
            _, eigvecs = torch.linalg.eigh(cov)
            # eigh returns eigenvalues in ASCENDING order; flip so
            # column 0 is the direction of maximum variance (matches
            # the standard PCA convention and pca_lowrank's ordering).
            rank = min(3, D)
            basis[:, :rank] = eigvecs[:, -rank:].flip(dims=[1]).to(dtype)
        except RuntimeError as exc:
            log_warn(f"PCA eigh failed ({exc}); using canonical basis.")

    if (
        prev_basis is not None
        and prev_basis.shape == basis.shape
        and torch.isfinite(prev_basis).all()
    ):
        prev = prev_basis.to(device=device, dtype=dtype)
        signs = torch.where((basis * prev).sum(dim=0) >= 0, 1.0, -1.0).to(basis)
        basis = basis * signs
    return basis


def pca_colorize_tensor(
    feats_flat: torch.Tensor,
    prev_basis: "torch.Tensor | None" = None,
    low_pct: float = 0.02,
    high_pct: float = 0.98,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit or reuse a 3-component PCA on ``(N, D)`` features and map to RGB.

    Returns ``(colors, basis)`` where ``colors`` is ``(N, 3)`` uint8
    and ``basis`` is ``(D, 3)`` float32. Non-finite rows are dropped
    from the fit and receive black in the output so bad inputs are
    visually obvious but don't poison the principal directions. When a
    compatible ``prev_basis`` is provided, it is reused instead of
    refitting so the expensive PCA/SVD-style solve happens only once.
    """
    flat = feats_flat.float()
    if flat.ndim != 2:
        raise ValueError(f"Expected feats_flat to have shape (N, D), got {tuple(flat.shape)}")

    N, D = flat.shape
    colors = torch.zeros((N, 3), device=flat.device, dtype=torch.uint8)

    valid_rows = torch.isfinite(flat).all(dim=1)
    #if not torch.all(valid_rows):
    #    log_warn("Skipping non-finite feature rows during PCA colorization.")
    valid = flat[valid_rows]

    centered = (
        valid - valid.mean(dim=0, keepdim=True) if valid.shape[0] > 0 else valid
    )
    if (
        prev_basis is not None
        and prev_basis.shape == (D, 3)
        and torch.isfinite(prev_basis).all()
    ):
        basis = prev_basis.to(device=flat.device, dtype=flat.dtype)
    else:
        basis = _pca_basis(centered)
    if valid.shape[0] == 0 or D == 0:
        return colors, basis

    proj = centered @ basis
    lo = torch.quantile(proj, low_pct, dim=0)
    hi = torch.quantile(proj, high_pct, dim=0)
    spread = hi - lo
    scaled = ((proj - lo) / spread.clamp(min=1e-6)).clamp(0.0, 1.0)
    # Degenerate axes (no spread) become mid-gray rather than an arbitrary
    # flat color that would wash out the visualization.
    scaled = torch.where(spread.unsqueeze(0) > 1e-6, scaled, torch.full_like(scaled, 0.5))
    colors[valid_rows] = (scaled * 255.0).to(torch.uint8)
    return colors, basis


def pca_colorize_with_basis(
    feats: torch.Tensor,
    prev_basis: "torch.Tensor | None" = None,
    low_pct: float = 0.02,
    high_pct: float = 0.98,
) -> Tuple[np.ndarray, torch.Tensor]:
    """Project ``(H, W, D)`` features to an ``(H, W, 3)`` uint8 image via PCA."""
    H, W, D = feats.shape
    colors, basis = pca_colorize_tensor(
        feats.reshape(-1, D), prev_basis=prev_basis, low_pct=low_pct, high_pct=high_pct
    )
    return colors.view(H, W, 3).cpu().numpy(), basis


def pca_colorize(
    feats: torch.Tensor,
    low_pct: float = 0.02,
    high_pct: float = 0.98,
) -> np.ndarray:
    """Project ``(H, W, D)`` features to an ``(H, W, 3)`` uint8 image via PCA."""
    image, _ = pca_colorize_with_basis(feats, low_pct=low_pct, high_pct=high_pct)
    return image


def upsample_nn(img_uint8: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor upsample ``(H, W, 3)`` uint8 to ``target_hw``."""
    t = torch.from_numpy(img_uint8).permute(2, 0, 1).unsqueeze(0).float()
    t = F.interpolate(t, size=target_hw, mode="nearest")
    return t[0].permute(1, 2, 0).to(torch.uint8).numpy()


def downsample_for_gui(img_uint8: np.ndarray, max_width: int = 320) -> np.ndarray:
    """Cheap preview downsample for viser GUI image widgets."""
    if img_uint8.shape[1] <= max_width:
        return img_uint8
    step = int(np.ceil(img_uint8.shape[1] / max_width))
    return np.ascontiguousarray(img_uint8[::step, ::step])


def _subsample_step(num_items: int, max_items: int) -> int:
    """Return a stride that keeps visualization uploads below ``max_items``."""
    if num_items <= max_items:
        return 1
    return int(np.ceil(num_items / max_items))


def show_empty_reconstruction(visualizer, voxel_size: float) -> None:
    """Publish empty RGB and feature-PCA point clouds to clear the viewer."""
    empty_xyz = np.zeros((0, 3), dtype=np.float32)
    empty_rgb = np.zeros((0, 3), dtype=np.uint8)
    rgb_handle = visualizer.add_point_cloud(
        pointcloud=empty_xyz,
        colors=empty_rgb,
        point_size=voxel_size,
        name="/reconstruction/rgb",
    )
    rgb_handle.visible = False
    visualizer.add_point_cloud(
        pointcloud=empty_xyz,
        colors=empty_rgb,
        point_size=voxel_size,
        name="/reconstruction/features_pca",
    )


def show_feature_reconstruction(
    visualizer,
    voxels,
    block_colors_pca: torch.Tensor,
    voxel_size: float,
) -> None:
    """Draw occupied voxels as RGB and feature-PCA point clouds in Viser."""
    voxel_colors_rgb = voxels.colors_uint8()
    voxel_colors_pca = block_colors_pca[voxels.block_idx_per_voxel]

    step = _subsample_step(len(voxels.centers), MAX_RECONSTRUCTION_POINTS)
    centers_np = voxels.centers[::step].cpu().numpy()
    voxel_colors_rgb_np = voxel_colors_rgb[::step].cpu().numpy()
    voxel_colors_pca_np = voxel_colors_pca[::step].cpu().numpy()

    rgb_handle = visualizer.add_point_cloud(
        pointcloud=centers_np,
        colors=voxel_colors_rgb_np,
        point_size=voxel_size,
        name="/reconstruction/rgb",
    )
    rgb_handle.visible = False
    visualizer.add_point_cloud(
        pointcloud=centers_np,
        colors=voxel_colors_pca_np,
        point_size=voxel_size,
        name="/reconstruction/features_pca",
    )


def process_frame(
    obs: CameraObservation,
    mapper: Mapper,
    feature_model: CRadioInference,
    depth_filter: FilterDepth,
    prev_pca_basis: "torch.Tensor | None" = None,
    surface_only: bool = True,
    extract_voxels: bool = False,
    timer: CudaEventTimer = None,
):
    """Integrate one RGB-D frame and optionally prepare visualization data.

    The mapper expects a batched :class:`~curobo.types.CameraObservation`, even
    when this tutorial uses one camera. This helper keeps the per-frame flow in
    one place: clean depth, extract C-RADIO features, integrate into the mapper,
    and optionally extract occupied voxels for the live PCA point cloud.

    Returns:
        ``(feats, voxels, block_colors_pca, pca_basis, tsdf_time_ms)``. ``feats``
        is the raw ``(H_p, W_p, D)`` RADIO patch map used for per-image PCA.
    """
    # 1. Depth prep: same filter as volumetric_mapping.py.
    inference_time = None
    obs.depth_image = torch.nan_to_num(obs.depth_image, nan=0.0)
    filtered, _ = depth_filter(obs.depth_image.unsqueeze(0))
    obs.depth_image = filtered[0]
    if timer is not None:
        timer.start()
    # 2. Neural image features: (H_p, W_p, D) on GPU.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feats = feature_model.extract_patch_features(obs.rgb_image)

    if timer is not None:
        inference_time = 1000 * timer.stop()

    # 3. Integrate: pack into a (num_cameras=1) batched observation so
    # the feature-integration kernel sees the expected (N, H_p, W_p, D)
    # channels-last layout.
    batched = CameraObservation(
        depth_image=obs.depth_image.unsqueeze(0),
        rgb_image=obs.rgb_image.unsqueeze(0),
        pose=Pose(
            position=obs.pose.position.view(1, 3),
            quaternion=obs.pose.quaternion.view(1, 4),
        ),
        intrinsics=obs.intrinsics.unsqueeze(0),
        feature_grid=feats.to(dtype=torch.float16).contiguous().unsqueeze(0),
    )

    tsdf_time_ms = None
    if timer is not None:
        timer.start()
    mapper.integrate(batched)
    if timer is not None:
        tsdf_time = timer.stop()
        tsdf_time_ms = 1000 * tsdf_time

    if not extract_voxels:
        return feats, None, None, prev_pca_basis, tsdf_time_ms, inference_time

    # 4. Extract occupied voxels from the current TSDF state.
    voxels = mapper.extract_occupied_voxels(surface_only=surface_only)
    if len(voxels) == 0:
        return feats, voxels, None, prev_pca_basis, tsdf_time_ms, inference_time

    # 5. PCA on per-block features, sign-aligned to the previous basis
    # so colors stay consistent across frames.
    block_features = voxels.block_data.features_normalized()
    block_colors_pca, pca_basis = pca_colorize_tensor(
        block_features, prev_basis=prev_pca_basis,
    )
    return feats, voxels, block_colors_pca, pca_basis, tsdf_time_ms, inference_time


def main():
    parser = argparse.ArgumentParser(description="Feature TSDF mapping on Sun3D")
    parser.add_argument("--root", type=str, required=True, help="Sun3D dataset root")
    parser.add_argument(
        "--num-frames", type=int, default=1000, help="Max number of frames to process"
    )
    parser.add_argument(
        "--stride", type=int, default=5, help="Frame stride through the sequence"
    )
    parser.add_argument("--voxel-size", type=float, default=0.025, help="Voxel size (m)")
    parser.add_argument(
        "--save-pca",
        action="store_true",
        help="Save per-frame RGB | PCA(features) panels for inspection",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open a viser server showing RGB + feature-PCA point clouds",
    )
    parser.add_argument(
        "--clear-aabb",
        type=float,
        nargs=6,
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
        default=None,
        help=(
            "Optional world-space AABB to clear once after integration. "
            "Clearing is conservative at block granularity and keeps blocks allocated."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print(f"Loading Sun3D from {args.root}...")
    dataset = Sun3dDataset(args.root, device=args.device)
    print(f"Found {len(dataset)} frames")

    text_adaptor = TEXT_ADAPTOR_NAME if args.visualize else None
    print(f"Loading C-RADIO ({RADIO_MODEL_NAME}) via NVlabs/RADIO torch.hub...")
    feature_model = CRadioInference(
        model_name=RADIO_MODEL_NAME,
        device=args.device,
        text_adaptor_name=text_adaptor,
    )
    text_model = feature_model if text_adaptor is not None else None
    if text_model is not None:
        print(f"Text matching enabled with adaptor='{feature_model.text_adaptor_name}'.")

    # Probe feature dim from the first frame so the mapper config stays in
    # sync with the model regardless of which C-RADIO variant is loaded.
    probe_feats = feature_model.extract_patch_features(dataset[0].rgb_image)
    feature_dim = probe_feats.shape[-1]
    print(f"Feature dim: {feature_dim}")
    print(f"Feature shape: {probe_feats.shape}")

    config = MapperCfg(
        voxel_size=args.voxel_size,
        extent_meters_xyz=MAP_EXTENT_METERS_XYZ,
        truncation_distance=args.voxel_size * 4,
        depth_maximum_distance=10.0,
        depth_minimum_distance=0.05,
        minimum_tsdf_weight=0.5,
        decay_factor=1.0,
        frustum_decay_factor=1.0,
        num_cameras=1,
        image_height=SUN3D_IMAGE_SHAPE[0],
        image_width=SUN3D_IMAGE_SHAPE[1],
        feature_dim=feature_dim,
        block_size=8,
    )
    mapper = Mapper(config)
    print(f"Mapper initialized: {mapper.memory_usage_mb():.1f} MB")

    depth_filter = FilterDepth(
        image_shape=SUN3D_IMAGE_SHAPE,
        depth_minimum_distance=mapper.config.depth_minimum_distance,
        depth_maximum_distance=mapper.config.depth_maximum_distance,
        flying_pixel_threshold=0.5,
        bilateral_kernel_size=3,
    )

    out_dir = Path(runtime.cache_dir) / "examples" / "feature_mapping"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Optional: set up viser before the loop so the scene grows in real time ---
    visualizer = None
    status = None
    text_status = None
    text_live_cb = None
    current_rgb_image = None
    current_feature_image = None
    run_text_match = None
    pca_basis = None
    live_state: dict = {
        "has_voxels": False,
        "view": None,
        "text_prompt": None,
        "text_query": None,
        "text_match": None,
    }

    if args.visualize:
        from curobo.viewer import ViserVisualizer

        visualizer = ViserVisualizer(connect_port=8080)
        server = visualizer._server
        print("Visualization: http://localhost:8080")

        with server.gui.add_folder("Mapping"):
            status = server.gui.add_text(
                "status", initial_value="Integrating...", disabled=True
            )
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            current_rgb_image = server.gui.add_image(
                blank, label="Current RGB", format="jpeg", jpeg_quality=80
            )
            current_feature_image = server.gui.add_image(
                blank, label="Current Feature PCA", format="jpeg", jpeg_quality=80
            )

        def clear_overlays(message: str | None = None):
            empty_xyz = np.zeros((0, 3), dtype=np.float32)
            empty_rgb = np.zeros((0, 3), dtype=np.uint8)
            visualizer.add_point_cloud(
                pointcloud=empty_xyz,
                colors=empty_rgb,
                point_size=args.voxel_size * 1.5,
                name="/reconstruction/text_matched",
            )
            live_state["text_match"] = None
            live_state["has_voxels"] = False
            if status is not None and message is not None:
                status.value = message
            if text_status is not None and message is not None:
                text_status.value = message

        def refresh_reconstruction(surface_only: bool = True):
            nonlocal pca_basis
            voxels_now = mapper.extract_occupied_voxels(
                surface_only=surface_only
            )
            live_state["has_voxels"] = len(voxels_now) > 0
            live_state["view"] = voxels_now.block_data

            if len(voxels_now) == 0:
                show_empty_reconstruction(visualizer, args.voxel_size)
                del voxels_now
                torch.cuda.empty_cache()
                return None

            block_features = voxels_now.block_data.features_normalized()
            block_colors_pca, pca_basis = pca_colorize_tensor(
                block_features, prev_basis=pca_basis,
            )
            show_feature_reconstruction(
                visualizer, voxels_now, block_colors_pca, args.voxel_size
            )
            del block_features, block_colors_pca, voxels_now
            torch.cuda.empty_cache()
            return None

        # --- Optional Text Matching panel ---
        if text_model is not None:
            with server.gui.add_folder("Text Matching"):
                text_input = server.gui.add_text("prompt", initial_value="table")
                text_top_k = server.gui.add_slider(
                    "top_k", min=10, max=2000, step=10, initial_value=TEXT_TOP_K
                )
                # Cosine in teacher (SigLIP-projected) space. Distribution
                # is much tighter than raw RADIO cosine (matches typically
                # land in ~0.05-0.30); -1 disables the cut.
                text_min_score = server.gui.add_slider(
                    "min_score", min=0.0, max=1.0, step=0.005,
                    initial_value=TEXT_MIN_SCORE,
                )
                text_live_cb = server.gui.add_checkbox(
                    "live_update", initial_value=True
                )
                text_search = server.gui.add_button("Search")
                text_clear = server.gui.add_button("Clear Matches")
                text_status = server.gui.add_text(
                    "text_status",
                    initial_value="Waiting for voxels...",
                    disabled=True,
                )

            def _clear_text_match():
                visualizer.add_point_cloud(
                    pointcloud=np.zeros((0, 3), dtype=np.float32),
                    colors=np.zeros((0, 3), dtype=np.uint8),
                    point_size=args.voxel_size * 1.5,
                    name="/reconstruction/text_matched",
                )

            def _get_live_text_query():
                prompt = text_input.value.strip()
                if not prompt:
                    live_state["text_prompt"] = None
                    live_state["text_query"] = None
                    return None, None
                if (
                    live_state["text_prompt"] != prompt
                    or live_state["text_query"] is None
                ):
                    live_state["text_prompt"] = prompt
                    live_state["text_query"] = text_model.encode_text(prompt)[0]
                return prompt, live_state["text_query"]

            def _current_text_min_score() -> Optional[float]:
                value = float(text_min_score.value)
                return value if value > -1.0 else None

            def run_text_match():
                view_now = live_state["view"]
                if view_now is None or not live_state["has_voxels"]:
                    text_status.value = "Waiting for voxels..."
                    return
                prompt, text_query = _get_live_text_query()
                if text_query is None:
                    _clear_text_match()
                    live_state["text_match"] = None
                    text_status.value = "Enter a prompt to highlight matches."
                    return
                min_score = _current_text_min_score()
                matched = mapper.extract_matching_feature_voxels(
                    feature_vector=text_query,
                    top_k=int(text_top_k.value),
                    surface_only=True,
                    minimum_score=min_score,
                    feature_projector=text_model.project_features,
                )
                if matched.block_pool_idx.numel() == 0:
                    _clear_text_match()
                    live_state["text_match"] = None
                    threshold_str = "off" if min_score is None else f"{min_score:.3f}"
                    text_status.value = (
                        f"text='{prompt}', no matches "
                        f"(min_score={threshold_str})"
                    )
                    del matched
                    torch.cuda.empty_cache()
                    return
                live_state["text_match"] = matched.block_pool_idx
                centers = matched.voxels.centers
                step = _subsample_step(len(centers), MAX_TEXT_MATCH_POINTS)
                highlight = np.zeros((len(centers[::step]), 3), dtype=np.uint8)
                highlight[:, 1] = 255
                highlight[:, 2] = 255
                matched_points_np = centers[::step].cpu().numpy()
                visualizer.add_point_cloud(
                    pointcloud=matched_points_np,
                    colors=highlight,
                    point_size=args.voxel_size * 1.5,
                    name="/reconstruction/text_matched",
                )
                top_score = float(matched.block_scores[0].item())
                threshold_str = "off" if min_score is None else f"{min_score:.3f}"
                text_status.value = (
                    f"text='{prompt}', #matched={len(centers)} "
                    f"({matched.block_scores.numel()} blocks, "
                    f"top score={top_score:.3f}, min_score={threshold_str})"
                )
                del matched, centers, matched_points_np
                torch.cuda.empty_cache()

            @text_search.on_click
            def _on_text_search(_):
                run_text_match()

            @text_clear.on_click
            def _on_text_clear(_):
                prompt, text_query = _get_live_text_query()
                if text_query is None:
                    _clear_text_match()
                    live_state["text_match"] = None
                    text_status.value = "Enter a prompt before clearing."
                    return
                if not live_state["has_voxels"]:
                    text_status.value = "Waiting for voxels..."
                    return
                # Reuse the cached block_pool_idx from the most recent
                # run_text_match if the prompt is unchanged; otherwise
                # rerun the projection with the current threshold.
                # block_pool_idx is already int32 in score-descending order,
                # so it can be passed straight to clear_blocks.
                cached = live_state["text_match"]
                if cached is None or live_state["text_prompt"] != prompt:
                    matched = mapper.extract_matching_feature_voxels(
                        feature_vector=text_query,
                        top_k=int(text_top_k.value),
                        surface_only=True,
                        minimum_score=_current_text_min_score(),
                        feature_projector=text_model.project_features,
                    )
                    cached = matched.block_pool_idx
                    live_state["text_match"] = cached
                    del matched
                    torch.cuda.empty_cache()
                if cached.numel() == 0:
                    _clear_text_match()
                    live_state["text_match"] = None
                    text_status.value = f"text='{prompt}', no blocks to clear"
                    return

                n_cleared = mapper.clear_blocks(cached)
                clear_overlays(f"Cleared {n_cleared} blocks for text='{prompt}'")
                refresh_reconstruction(surface_only=True)

            @text_top_k.on_update
            def _on_text_top_k(_):
                run_text_match()

            @text_min_score.on_update
            def _on_text_min_score(_):
                run_text_match()

            @text_live_cb.on_update
            def _on_text_live(_):
                if text_live_cb.value:
                    run_text_match()

    # --- Main loop: per-frame pipeline ---
    indices = list(range(0, len(dataset), args.stride))[: args.num_frames]
    pbar = tqdm(indices, desc="integrating")
    timer = CudaEventTimer()
    for idx in pbar:
        obs = dataset[idx]

        feats, voxels, block_colors_pca, pca_basis, tsdf_time_ms, inference_time = process_frame(
            obs, mapper, feature_model, depth_filter, prev_pca_basis=pca_basis,
            extract_voxels=visualizer is not None,
            timer=timer,
        )

        current_feature_pca_img = None
        if current_rgb_image is not None and current_feature_image is not None:
            rgb_np = obs.rgb_image.cpu().numpy()
            current_feature_pca_img, pca_basis = pca_colorize_with_basis(
                feats, prev_basis=pca_basis
            )
            feature_img = upsample_nn(current_feature_pca_img, tuple(rgb_np.shape[:2]))
            current_rgb_image.image = downsample_for_gui(rgb_np)
            current_feature_image.image = downsample_for_gui(feature_img)

        if args.save_pca:
            if current_feature_pca_img is None:
                current_feature_pca_img, pca_basis = pca_colorize_with_basis(
                    feats, prev_basis=pca_basis
                )
            pca_up_np = upsample_nn(
                current_feature_pca_img, tuple(obs.rgb_image.shape[:2])
            )
            panel = np.concatenate([obs.rgb_image.cpu().numpy(), pca_up_np], axis=1)
            iio.imwrite(out_dir / f"pca_{idx:06d}.png", panel)

        num_blocks = 0
        if visualizer is not None and voxels is not None:
            num_blocks = voxels.block_data.num_allocated
            live_state["has_voxels"] = len(voxels) > 0
            live_state["view"] = voxels.block_data

            if block_colors_pca is not None:
                show_feature_reconstruction(
                    visualizer, voxels, block_colors_pca, args.voxel_size
                )
                if status is not None:
                    status.value = f"frame={idx}, blocks={num_blocks}"
                if (
                    run_text_match is not None
                    and text_live_cb is not None
                    and text_live_cb.value
                ):
                    run_text_match()

            del voxels, block_colors_pca
            torch.cuda.empty_cache()

        postfix = {"blocks": num_blocks}
        if tsdf_time_ms is not None:
            postfix["tsdf_ms"] = f"{tsdf_time_ms:.3f}"
        if inference_time is not None:
            postfix["infer_ms"] = f"{inference_time:.3f}"
        pbar.set_postfix(postfix)

    if args.clear_aabb is not None:
        clear_min = torch.tensor(args.clear_aabb[:3], device=args.device, dtype=torch.float32)
        clear_max = torch.tensor(args.clear_aabb[3:], device=args.device, dtype=torch.float32)
        n_cleared = mapper.clear_region(clear_min, clear_max)
        print(
            "Cleared "
            f"{n_cleared} allocated blocks intersecting AABB "
            f"{clear_min.cpu().tolist()} -> {clear_max.cpu().tolist()}"
        )

        if visualizer is not None:
            clear_overlays(f"Cleared {n_cleared} blocks")
            refresh_reconstruction(surface_only=True)

    print(f"\nMapper memory: {mapper.memory_usage_mb():.1f} MB")
    if args.save_pca:
        print(f"PCA panels saved to: {out_dir}")

    if visualizer is not None:
        if text_model is not None:
            print("Integration done. Use the Text Matching panel. Ctrl+C to exit.")
        else:
            print("Integration done. Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
