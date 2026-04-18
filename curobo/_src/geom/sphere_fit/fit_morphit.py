# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""MorphIt sphere fitting -- reimplementation based on Nechyporenko et al.

Packs a set of optimisable spheres inside a triangle mesh by minimising
four losses (coverage, outside-volume, tangency, overlap) with Adam and
optional adaptive density control.

Sphere centres and radii are initialised externally (e.g. via
:func:`~curobo._src.geom.sphere_fit.fit_voxel.voxel_fit_mesh`) and
passed in as ``init_centers`` / ``init_radii``.

Public entry point: :func:`morphit_sphere_fit`.

Reference:

    Nechyporenko, N., Zhang, Y., Campbell, S., & Roncone, A. (2025).
    *MorphIt: Flexible Spherical Approximation of Robot Morphology for
    Representation-driven Adaptation.* arXiv:2507.14061.
"""

# Standard Library
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Third Party
import numpy as np
import torch
import torch.nn as nn
import trimesh

# CuRobo
from curobo._src.geom.sphere_fit.wp_mesh_query import WarpMeshQuery, WarpSphereSDFFunction
from curobo._src.util.logging import log_info, log_warn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MorphItLossWeights:
    """Loss weights for MorphIt optimization.

    Five loss terms:

    - **coverage** + **protrusion**: the core coverage-vs-protrusion pair.
    - **tangency**: encourages spheres to touch the mesh surface.
    - **overlap**: penalises pairwise sphere-sphere intersection.
    - **halfplane**: penalises spheres that cross a user-specified clip plane.
    """

    #: Penalises interior sample points not covered by any sphere.
    #: Higher values force spheres to fill the mesh volume more completely.
    coverage: float = 1000.0

    #: Protrusion loss: penalises sphere surface area that lies outside the
    #: mesh, estimated via Monte-Carlo SDF queries on the sphere surface.
    #: Differentiable w.r.t. both centres and radii.
    protrusion: float = 10.0

    #: Tangency loss: penalises gap between sphere surface and nearest mesh
    #: surface (encourages radius to match |SDF(center)|).
    tangency: float = 1.0

    #: Penalises pairwise sphere-sphere overlap.
    #: Keeps spheres from wasting budget by covering the same region.
    overlap: float = 0.1

    #: Penalises spheres that extend past a user-specified clip plane
    #: (e.g. to keep base-link spheres from protruding into a mounting surface).
    #: Only active when a clip plane is provided.
    halfplane: float = 1000.0

    #: Number of points sampled per sphere for the protrusion loss.
    protrusion_samples: int = 128


@dataclass
class MorphItConfig:
    """Configuration for the MorphIt sphere-fitting optimizer."""

    #: Number of spheres to initialise.
    num_spheres: int = 25

    #: Torch device for optimization tensors.
    device: torch.device = torch.device("cuda", 0)

    #: Number of interior sample points for coverage loss.
    num_inside_samples: int = 1000

    # -- Training -------------------------------------------------------------
    #: Total optimization iterations.
    iterations: int = 200

    #: Adam learning rate for sphere centres.
    center_lr: float = 0.005

    #: Adam learning rate for sphere radii.
    radius_lr: float = 0.001

    #: Maximum gradient norm for clipping.
    grad_clip_norm: float = 1.0

    #: Loss weights for each loss component.
    loss_weights: MorphItLossWeights = field(default_factory=MorphItLossWeights)

    # -- Density control ------------------------------------------------------
    #: Minimum iterations between density-control operations.
    density_control_interval: int = 20

    #: Minimum sphere radius as fraction of mesh extent; smaller spheres are pruned.
    radius_threshold_ratio: float = 0.01

    #: Coverage distance threshold as fraction of mesh extent for adding new spheres.
    coverage_threshold_ratio: float = 0.05

    #: Maximum allowed sphere count.
    max_spheres: int = 0  # 0 means num_spheres (set in post-init)

    # -- Clip plane -----------------------------------------------------------
    #: Half-plane constraint ``(normal, offset)`` in link-local coordinates.
    #: Spheres are penalised when ``dot(normal, center) - radius < offset``.
    #: When ``None``, the constraint is inactive.
    clip_plane: Optional[Tuple[Tuple[float, float, float], float]] = None

    #: Buffer distance (metres) added to the clip-plane constraint so spheres
    #: clear the plane by at least this margin.  Matches the default collision
    #: activation distance used by the motion planner.
    clip_plane_buffer: float = 0.02

    # -- Logging --------------------------------------------------------------
    #: Print progress every N iterations (0 = silent).
    verbose_frequency: int = 0

    def __post_init__(self):
        if self.max_spheres == 0:
            self.max_spheres = max(self.num_spheres, 300)

    def get_radius_threshold(self, mesh: trimesh.Trimesh) -> float:
        """Compute absolute radius threshold from mesh extent."""
        extent = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        return self.radius_threshold_ratio * extent

    def get_coverage_threshold(self, mesh: trimesh.Trimesh) -> float:
        """Compute absolute coverage threshold from mesh extent."""
        extent = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        return self.coverage_threshold_ratio * extent


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

#@get_torch_jit_decorator()
def _coverage_loss(
    inside_samples: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    mesh_extent: float,
) -> torch.Tensor:
    """Coverage loss: penalise interior points not covered by any sphere.

    Uses a soft-min (log-sum-exp) so all nearby spheres receive gradients,
    combined with a squared penalty to prioritise the worst-covered regions.
    Normalised by mesh extent squared for scale-invariance.

    Args:
        inside_samples: ``(M, 3)`` points inside the mesh.
        centers: ``(N, 3)`` sphere centres.
        radii: ``(N,)`` sphere radii.
        mesh_extent: Bounding-box diagonal of the mesh.

    Returns:
        Scalar loss.
    """
    if len(inside_samples) == 0 or len(centers) == 0:
        return torch.tensor(0.0, device=centers.device)
    dists = torch.cdist(inside_samples, centers)  # (M, N)
    coverage = dists - radii.unsqueeze(0)
    temperature = mesh_extent * 0.01
    soft_min = -temperature * torch.logsumexp(-coverage / temperature, dim=1)
    return torch.mean(torch.relu(soft_min) ** 2) / (mesh_extent ** 2)


#@get_torch_jit_decorator()
def _overlap_penalty(
    centers: torch.Tensor, radii: torch.Tensor, mesh_extent: float,
) -> torch.Tensor:
    """Overlap penalty: penalise pairwise sphere-sphere intersection.

    Normalised by mesh extent for scale-invariance.

    Args:
        centers: ``(N, 3)`` sphere centres.
        radii: ``(N,)`` sphere radii.
        mesh_extent: Bounding-box diagonal of the mesh.

    Returns:
        Scalar loss.
    """
    n = centers.shape[0]
    dists = torch.cdist(centers, centers)  # (N, N)
    dists = dists + torch.eye(n, device=centers.device) * 1e6
    radii_sum = radii.unsqueeze(1) + radii.unsqueeze(0)
    return torch.mean(torch.relu(radii_sum - dists)) / mesh_extent


def _sample_sphere_directions(n_samples: int, device: torch.device) -> torch.Tensor:
    """Generate uniformly distributed directions on the unit sphere.

    Args:
        n_samples: Number of direction vectors to generate.
        device: Torch device.

    Returns:
        ``(n_samples, 3)`` unit direction vectors.
    """
    phi = torch.rand(n_samples, device=device) * 2 * 3.14159265
    cos_theta = torch.rand(n_samples, device=device) * 2 - 1
    sin_theta = torch.sqrt(1 - cos_theta ** 2)
    return torch.stack([
        sin_theta * torch.cos(phi),
        sin_theta * torch.sin(phi),
        cos_theta,
    ], dim=1)


def _protrusion_loss(
    centers: torch.Tensor,
    radii: torch.Tensor,
    mesh_query,
    sdf_fn,
    directions: torch.Tensor,
    mesh_extent: float,
) -> torch.Tensor:
    """Protrusion loss: penalise sphere surface area lying outside the mesh.

    Projects pre-computed unit-sphere *directions* onto each sphere's surface,
    queries the mesh SDF differentiably at those points, and penalises positive
    SDF values (outside).  The penalty is ``relu(sdf)^2`` so the gradient is
    stronger for deeper protrusions.  Fully differentiable w.r.t. centres and
    radii.

    Args:
        centers: ``(N, 3)`` sphere centres.
        radii: ``(N,)`` sphere radii.
        mesh_query: :class:`WarpMeshQuery` instance.
        sdf_fn: Differentiable SDF function (``WarpSphereSDFFunction``).
        directions: ``(K, 3)`` fixed unit-sphere directions.
        mesh_extent: Bounding-box diagonal for normalisation.

    Returns:
        Scalar loss.
    """
    num_spheres = centers.shape[0]
    # Broadcast: (N, K, 3) = (N, 1, 3) + (N, 1, 1) * (1, K, 3)
    surface_pts = centers.unsqueeze(1) + radii.unsqueeze(1).unsqueeze(2) * directions.unsqueeze(0)
    surface_pts_flat = surface_pts.reshape(-1, 3)  # (N*K, 3)

    sdf_vals = sdf_fn.apply(surface_pts_flat, mesh_query)  # (N*K,)
    outside_penalty = torch.relu(sdf_vals) ** 2
    per_sphere = outside_penalty.reshape(num_spheres, -1).max(dim=1).values
    temperature = mesh_extent * 0.01
    soft_max = temperature * torch.logsumexp(per_sphere / temperature, dim=0)
    return soft_max / (mesh_extent ** 2)


def _tangency_loss(
    centers: torch.Tensor,
    radii: torch.Tensor,
    mesh_query,
    sdf_fn,
    mesh_extent: float,
) -> torch.Tensor:
    """Tangency loss: encourage each sphere to touch the mesh surface.

    Penalises spheres whose radius is smaller than ``|SDF(center)|``, i.e.,
    there is a gap between the sphere surface and the nearest mesh surface.
    Protrusion (radius > |SDF|) is handled by the outside-volume loss.

    Uses a single differentiable SDF query at the N sphere centres, replacing
    the more expensive SQEM formulation that required S surface samples.

    Args:
        centers: ``(N, 3)`` sphere centres.
        radii: ``(N,)`` sphere radii.
        mesh_query: :class:`WarpMeshQuery` instance.
        sdf_fn: Differentiable SDF function (``WarpSphereSDFFunction``).
        mesh_extent: Bounding-box diagonal for normalisation.

    Returns:
        Scalar loss.
    """
    sdf = sdf_fn.apply(centers, mesh_query)
    gap = torch.abs(sdf) - radii
    return torch.mean(torch.relu(gap) ** 2) / (mesh_extent ** 2)


def _halfplane_loss(
    centers: torch.Tensor,
    radii: torch.Tensor,
    plane_normal: torch.Tensor,
    plane_offset: float,
    buffer: float = 0.02,
) -> torch.Tensor:
    """Half-plane loss: penalise spheres that cross a clip plane.

    The signed clearance of each sphere to the plane is
    ``dot(normal, center) - offset - radius - buffer``.  Negative clearance
    means the sphere extends past the plane (or within the buffer zone) and
    is penalised with a squared hinge.

    Unlike the other losses, this is *not* normalised by mesh extent because
    the constraint is absolute (metres), not relative to mesh size.

    Args:
        centers: ``(N, 3)`` sphere centres.
        radii: ``(N,)`` sphere radii.
        plane_normal: ``(3,)`` unit normal pointing into the *allowed* half-space.
        plane_offset: Signed offset of the plane along the normal.
        buffer: Minimum clearance (metres) between sphere boundary and plane.

    Returns:
        Scalar loss.
    """
    signed_dist = centers @ plane_normal - plane_offset
    clearance = signed_dist - radii - buffer
    return torch.mean(torch.relu(-clearance) ** 2)


# ---------------------------------------------------------------------------
# Density control
# ---------------------------------------------------------------------------

def _prune_spheres(
    centers: nn.Parameter,
    radii: nn.Parameter,
    mesh_query,
    radius_threshold: float,
    device: torch.device,
) -> Tuple[nn.Parameter, nn.Parameter, int]:
    """Remove spheres that are too small to be useful.

    Only prunes spheres below *radius_threshold*.  Mostly-outside spheres
    are kept during training so the optimizer can still recover them; they
    are handled by the final prune after training completes.

    Args:
        centers: ``(N, 3)`` sphere centres.
        radii: ``(N,)`` sphere radii.
        mesh_query: :class:`WarpMeshQuery` instance.
        radius_threshold: Minimum radius.
        device: Torch device.

    Returns:
        Tuple ``(new_centers, new_radii, num_removed)``.
    """
    with torch.no_grad():
        prune = radii < radius_threshold
        valid = ~prune
        if not valid.any():
            best = radii.argmax()
            valid[best] = True
            prune[best] = False
        removed = int(prune.sum().item())
    if removed > 0:
        centers = nn.Parameter(centers[valid].detach().clone())
        radii = nn.Parameter(radii[valid].detach().clone())
    return centers, radii, removed


def _add_spheres(
    centers: nn.Parameter,
    radii: nn.Parameter,
    inside_samples: torch.Tensor,
    mesh_query,
    coverage_threshold: float,
    max_spheres: int,
    device: torch.device,
) -> Tuple[nn.Parameter, nn.Parameter, int]:
    """Add spheres to poorly-covered interior regions.

    Returns the inputs unchanged when *centers* or *inside_samples* are empty.

    Args:
        centers: ``(N, 3)`` sphere centres.
        radii: ``(N,)`` sphere radii.
        inside_samples: ``(M, 3)`` interior sample points.
        mesh_query: :class:`WarpMeshQuery` for SDF radius initialisation.
        coverage_threshold: Distance threshold for poor coverage.
        max_spheres: Maximum allowed total spheres.
        device: Torch device.

    Returns:
        Tuple ``(new_centers, new_radii, num_added)``.
    """
    if len(centers) == 0 or len(inside_samples) == 0:
        return centers, radii, 0

    with torch.no_grad():
        dists = torch.cdist(inside_samples, centers)
        coverage = dists - radii.unsqueeze(0)
        min_cov, _ = torch.min(coverage, dim=1)

        poor_mask = min_cov > coverage_threshold
        poor_pts = inside_samples[poor_mask]

        space = max_spheres - len(radii)
        to_add = min(len(poor_pts), space)
        if to_add <= 0:
            return centers, radii, 0

        scores = min_cov[poor_mask]
        sorted_idx = torch.argsort(scores, descending=True)

        repel = float(radii.min()) * 0.5
        selected: List[int] = []
        for idx in sorted_idx:
            if len(selected) >= to_add:
                break
            p = poor_pts[idx]
            d_existing = torch.norm(centers - p.unsqueeze(0), dim=1)
            if torch.min(d_existing) < repel:
                continue
            if selected:
                chosen = poor_pts[torch.tensor(selected, device=device)]
                if torch.min(torch.norm(chosen - p.unsqueeze(0), dim=1)) < repel:
                    continue
            selected.append(idx.item())

        if len(selected) == 0:
            return centers, radii, 0

        new_c = poor_pts[torch.tensor(selected, device=device)]
        sdf_vals, _ = mesh_query.query_sdf(new_c.contiguous())
        new_r = torch.abs(sdf_vals).clamp(min=1e-4)
        centers = nn.Parameter(torch.cat([centers.detach(), new_c], dim=0))
        radii = nn.Parameter(torch.cat([radii.detach(), new_r], dim=0))
    return centers, radii, len(selected)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _build_optimizer(
    centers: nn.Parameter, radii: nn.Parameter, cfg: MorphItConfig
) -> torch.optim.Adam:
    """Create an Adam optimizer for sphere centres and radii."""
    return torch.optim.Adam([
        {"params": centers, "lr": cfg.center_lr},
        {"params": radii, "lr": cfg.radius_lr},
    ])


def _run_training(
    mesh: trimesh.Trimesh,
    cfg: MorphItConfig,
    init_centers: torch.Tensor,
    init_radii: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """Core MorphIt training loop.

    Args:
        mesh: Input triangle mesh.
        cfg: :class:`MorphItConfig`.
        init_centers: ``(N, 3)`` initial sphere centres on *device*.
        init_radii: ``(N,)`` initial sphere radii on *device*.

    Returns:
        Tuple ``(centers, radii, history)`` where *history* is a list of
        ``(centers, radii)`` numpy snapshots recorded after each iteration.
    """
    radius_threshold = cfg.get_radius_threshold(mesh)
    coverage_threshold = cfg.get_coverage_threshold(mesh)

    device = cfg.device

    centers_param = nn.Parameter(init_centers.to(device).float())
    radii_param = nn.Parameter(init_radii.to(device).float())
    log_info(
        f"[MorphIt init] seeds: {centers_param.shape[0]} spheres, "
        f"radii range [{radii_param.min().item():.4f}, {radii_param.max().item():.4f}]"
    )

    if cfg.num_spheres <= 0 and centers_param.shape[0] > 0:
        cfg.max_spheres = int(centers_param.shape[0] * 1.5)
        log_info(f"[MorphIt] auto num_spheres={centers_param.shape[0]}, "
                 f"max_spheres={cfg.max_spheres}")

    if centers_param.shape[0] == 0:
        log_warn("[MorphIt] no initial spheres, returning empty result")
        return np.zeros((0, 3)), np.zeros((0,)), []

    # -- Warp mesh query (for SDF losses + containment) -----------------------
    mesh_query = WarpMeshQuery(mesh, device)

    # -- Initialise sample points ---------------------------------------------
    parts: List[np.ndarray] = []

    # Volume samples: uniform interior points
    n_volume = cfg.num_inside_samples // 2
    try:
        vol_pts = trimesh.sample.volume_mesh(mesh, count=n_volume * 2)
        if len(vol_pts) > 0:
            parts.append(vol_pts[:n_volume])
    except Exception as e:
        log_warn(f"morphit: volume_mesh sampling failed ({e}), using surface samples only")

    # Surface-inset samples: surface points pushed inward along face normals
    n_surface = cfg.num_inside_samples - n_volume
    surf_pts, face_ids = trimesh.sample.sample_surface(mesh, n_surface)
    face_normals_np = mesh.face_normals[face_ids]
    inset = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])) * 0.005
    inset_pts = surf_pts - face_normals_np * inset

    inset_t = torch.tensor(inset_pts, dtype=torch.float32, device=device).contiguous()
    with torch.no_grad():
        sdf_check, _ = mesh_query.query_sdf(inset_t)
        inside_mask = sdf_check < 0
    inset_inside = inset_pts[inside_mask.cpu().numpy()]
    if len(inset_inside) > 0:
        parts.append(inset_inside)

    if len(parts) == 0:
        log_warn("[MorphIt] no interior samples, returning init spheres directly")
        return (
            centers_param.detach().cpu().numpy(),
            radii_param.detach().cpu().numpy(),
            [],
        )

    all_pts = np.vstack(parts)
    inside_samples = torch.tensor(all_pts, dtype=torch.float32, device=device)

    # -- Mesh extent for loss normalisation -----------------------------------
    mesh_extent = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    bbox_dims = mesh.bounds[1] - mesh.bounds[0]
    max_radius = float(np.min(bbox_dims)) / 2.0
    log_info(f"[MorphIt] mesh_extent={mesh_extent:.4f}  max_radius={max_radius:.4f}")

    # -- Clip plane tensors (if provided) -------------------------------------
    hp_normal: Optional[torch.Tensor] = None
    hp_offset: float = 0.0
    if cfg.clip_plane is not None:
        normal_tuple, hp_offset = cfg.clip_plane
        hp_normal = torch.tensor(normal_tuple, dtype=torch.float32, device=device)
        hp_normal = hp_normal / hp_normal.norm()
        log_info(f"[MorphIt] clip plane: normal={normal_tuple}, offset={hp_offset}")

    # -- Optimizer & weights --------------------------------------------------
    optimizer = _build_optimizer(centers_param, radii_param, cfg)
    w = cfg.loss_weights
    prot_dirs = _sample_sphere_directions(w.protrusion_samples, device)

    # -- Training loop --------------------------------------------------------
    history: List[Tuple[np.ndarray, np.ndarray]] = []
    last_dc_iter = 0

    for it in range(cfg.iterations):
        optimizer.zero_grad()

        # -- Compute losses (normalised by mesh extent) -----------------------
        loss_cov = _coverage_loss(inside_samples, centers_param, radii_param, mesh_extent)
        loss_prot = _protrusion_loss(
            centers_param, radii_param, mesh_query, WarpSphereSDFFunction,
            prot_dirs, mesh_extent,
        )
        loss_tan = _tangency_loss(
            centers_param, radii_param, mesh_query, WarpSphereSDFFunction,
            mesh_extent,
        )
        loss_ovl = _overlap_penalty(centers_param, radii_param, mesh_extent)

        loss_hp = torch.tensor(0.0, device=device)
        if hp_normal is not None:
            loss_hp = _halfplane_loss(
                centers_param, radii_param, hp_normal, hp_offset,
                buffer=cfg.clip_plane_buffer,
            )

        total = (
            w.coverage * loss_cov
            + w.protrusion * loss_prot
            + w.tangency * loss_tan
            + w.overlap * loss_ovl
            + w.halfplane * loss_hp
        )

        # -- Backward + step --------------------------------------------------
        total.backward()
        torch.nn.utils.clip_grad_norm_(centers_param, cfg.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(radii_param, cfg.grad_clip_norm)
        optimizer.step()

        with torch.no_grad():
            radii_param.clamp_(min=1e-6)

        # -- Logging ----------------------------------------------------------
        if cfg.verbose_frequency > 0 and it % cfg.verbose_frequency == 0:
            hp_str = f" hp={loss_hp.item():.6f}" if hp_normal is not None else ""
            log_info(
                f"[MorphIt iter {it}] loss={total.item():.6f}  "
                f"cov={loss_cov.item():.4f} prot={loss_prot.item():.6f} "
                f"tan={loss_tan.item():.6f} "
                f"ovl={loss_ovl.item():.4f}"
                f"{hp_str} "
                f"num_spheres={centers_param.shape[0]}"
            )

        # -- Density control ---------------------------------------------------
        if (
            cfg.density_control_interval > 0
            and it - last_dc_iter >= cfg.density_control_interval
            and it > 0
        ):
            centers_param, radii_param, removed = _prune_spheres(
                centers_param, radii_param, mesh_query, radius_threshold, device,
            )
            centers_param, radii_param, added = _add_spheres(
                centers_param, radii_param, inside_samples, mesh_query,
                coverage_threshold, cfg.max_spheres, device,
            )
            if removed > 0 or added > 0:
                optimizer = _build_optimizer(centers_param, radii_param, cfg)
                log_info(
                    f"[MorphIt DC iter {it}] removed={removed} added={added} "
                    f"total={centers_param.shape[0]}"
                )
            last_dc_iter = it

        # -- Record snapshot ---------------------------------------------------
        with torch.no_grad():
            history.append((
                centers_param.detach().cpu().numpy().copy(),
                radii_param.detach().cpu().numpy().copy(),
            ))

    # -- Final prune: remove spheres that are too small to contribute ----------
    with torch.no_grad():
        small = radii_param < radius_threshold
        if small.any():
            valid = ~small
            if not valid.any():
                best = radii_param.argmax()
                valid[best] = True
            log_info(
                f"[MorphIt] final prune: removed {int((~valid).sum().item())} small spheres, "
                f"{int(valid.sum().item())} remaining"
            )
            centers_param = nn.Parameter(centers_param[valid].detach().clone())
            radii_param = nn.Parameter(radii_param[valid].detach().clone())
            history.append((
                centers_param.detach().cpu().numpy().copy(),
                radii_param.detach().cpu().numpy().copy(),
            ))

    return (
        centers_param.detach().cpu().numpy(),
        radii_param.detach().cpu().numpy(),
        history,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def morphit_sphere_fit(
    mesh: trimesh.Trimesh,
    num_spheres: Optional[int] = None,
    iterations: int = 200,
    max_attempts: int = 10,
    loss_weights: Optional[MorphItLossWeights] = None,
    device: torch.device = torch.device("cuda", 0),
    init_centers: np.ndarray = None,
    init_radii: np.ndarray = None,
    max_spheres: int = 0,
    clip_plane: Optional[Tuple[Tuple[float, float, float], float]] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
    """Fit spheres to a mesh using the MorphIt optimization algorithm.

    Callers must provide initial sphere seeds via *init_centers* and
    *init_radii*.  Use :func:`voxel_fit_mesh` to generate seeds.

    Args:
        mesh: Input triangle mesh.
        num_spheres: Desired number of spheres.  When ``None``, the count is
            inferred from the length of *init_centers*.
        iterations: Number of optimization iterations.
        max_attempts: Retry count if optimization returns fewer than 2 spheres.
        loss_weights: Loss weights. Defaults to :class:`MorphItLossWeights` defaults.
        device: Torch device for optimization tensors.
        init_centers: ``(N, 3)`` initial sphere centres as numpy array.
        init_radii: ``(N,)`` initial sphere radii as numpy array.
        max_spheres: Maximum sphere count allowed during density control.
            When 0 (default), uses *num_spheres*.
        clip_plane: Half-plane constraint ``(normal, offset)`` in mesh-local
            coordinates.  Spheres are penalised (and hard-clamped after
            training) so they do not extend past the plane.  ``normal``
            points into the allowed half-space.  When ``None`` (default),
            no clipping is applied.

    Returns:
        Tuple ``(centers, radii, history)`` where *centers* is ``(N, 3)``,
        *radii* is ``(N,)``, and *history* is a per-iteration list of
        ``(centers, radii)`` numpy snapshots.  Returns empty arrays if all
        attempts produce fewer than 2 spheres.
    """
    if init_centers is None or init_radii is None:
        raise ValueError("morphit_sphere_fit requires init_centers and init_radii")

    log_info(
        f"[MorphIt] mesh: {len(mesh.vertices)} verts, "
        f"{len(mesh.faces)} faces, vol={mesh.volume:.4e}"
    )

    num = num_spheres if num_spheres is not None else len(init_centers)
    effective_max = max_spheres if max_spheres > 0 else min(int(num * 2), 50)

    cfg = MorphItConfig(
        num_spheres=num,
        iterations=iterations,
        max_spheres=effective_max,
        loss_weights=loss_weights or MorphItLossWeights(),
        device=device,
        verbose_frequency=max(1, iterations // 10),
        density_control_interval=20,
        clip_plane=clip_plane,
    )

    init_c_t = torch.as_tensor(init_centers, dtype=torch.float32, device=device)
    init_r_t = torch.as_tensor(init_radii, dtype=torch.float32, device=device)

    n_pts: Optional[np.ndarray] = None
    n_radius: Optional[np.ndarray] = None
    history: List[Tuple[np.ndarray, np.ndarray]] = []

    for attempt in range(max_attempts):
        n_pts, n_radius, history = _run_training(mesh, cfg, init_c_t, init_r_t)
        if n_pts is not None and len(n_pts) > 1:
            break
        log_warn(f"morphit_sphere_fit: attempt {attempt} returned too few spheres")

    return n_pts, n_radius, history
