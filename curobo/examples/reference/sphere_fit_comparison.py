# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sphere Fitting Comparison Demo -- visualise different fitting methods in Viser.

Loads a robot URDF, picks a subset of collision links, fits spheres to each link
using every :class:`SphereFitType` method, and renders the results side-by-side
in an interactive Viser viewer.

Each fitting method is displayed as a column of coloured spheres next to the
original mesh, so you can visually compare coverage and density.

Usage::

    python sphere_fit_comparison.py
    python sphere_fit_comparison.py --urdf franka.yml
    python sphere_fit_comparison.py --links panda_link0 panda_link7 panda_hand
    python sphere_fit_comparison.py --port 8081

Press Ctrl-C to stop the server.
"""

# Standard Library
from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Tuple

# Third Party
import numpy as np
import trimesh
import viser

# CuRobo
from curobo.config_io import join_path, load_yaml
from curobo.content import get_assets_path, get_robot_configs_path
from curobo.logging import log_warn, setup_logger
from curobo.robot_parser import UrdfRobotParser
from curobo.scene import Sphere
from curobo.sphere_fit import (
    SphereFitType,
    estimate_sphere_count,
    fit_spheres_to_mesh,
)

# ---------------------------------------------------------------------------
# Colour palette -- one colour per fitting method
# ---------------------------------------------------------------------------

METHOD_COLORS: Dict[SphereFitType, Tuple[int, int, int]] = {
    SphereFitType.SURFACE: (30, 144, 255),                # dodger blue
    SphereFitType.VOXEL: (50, 205, 50),                  # lime green
    SphereFitType.MORPHIT: (255, 105, 180),              # hot pink
}

# Horizontal offset between columns (one per method)
COLUMN_SPACING = 0.4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_parser_from_yml(
    config_path: str,
) -> Tuple[UrdfRobotParser, List[str]]:
    """Load a URDF parser from a cuRobo YAML config file.

    Returns the parser and the list of link names that have geometry.
    """
    robot_configs_path = get_robot_configs_path()
    full_path = join_path(robot_configs_path, config_path)
    assets_path = get_assets_path()
    data = load_yaml(full_path)
    if "robot_cfg" in data:
        data = data["robot_cfg"]
    kin_data = data["kinematics"]
    urdf_path = join_path(assets_path, kin_data["urdf_path"])
    asset_path = join_path(assets_path, kin_data.get("asset_root_path", ""))

    parser = UrdfRobotParser(
        urdf_path, load_meshes=True, mesh_root=asset_path, build_scene_graph=True,
    )
    parser.build_link_parent()

    all_links = parser.get_link_names_from_urdf()
    mesh_links = [
        ln for ln in all_links if len(parser.get_link_geometry(ln)) > 0
    ]
    return parser, mesh_links


def _get_link_mesh(
    parser: UrdfRobotParser, link_name: str,
) -> Optional[trimesh.Trimesh]:
    """Get the combined trimesh for a link, preprocessed for sphere fitting."""
    geom_list = parser.get_link_geometry(link_name)
    if not geom_list:
        return None

    meshes = []
    for geom in geom_list:
        m = geom.get_trimesh_mesh(transform_with_pose=True)
        if m is not None:
            m.fill_holes()
            trimesh.repair.fix_normals(m)
            trimesh.repair.fix_inversion(m)
            trimesh.repair.fix_winding(m)
            meshes.append(m)

    if not meshes:
        return None
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _add_spheres_to_viser(
    server: viser.ViserServer,
    positions: np.ndarray,
    radii: np.ndarray,
    color: Tuple[int, int, int],
    name: str,
    offset: np.ndarray,
) -> None:
    """Render a set of spheres as batched meshes with a spatial offset."""
    if positions is None or len(positions) == 0:
        return

    radii = radii.copy()
    radii[radii < 0.001] = 0.001

    # Create a unit sphere mesh for instancing
    ref_sphere = Sphere(
        name="ref", pose=[0, 0, 0, 1, 0, 0, 0], radius=float(radii[0]),
    ).get_trimesh_mesh()

    scale = radii / radii[0]
    shifted_pos = positions + offset
    quats = np.zeros((len(positions), 4))
    quats[:, 0] = 1.0
    colors = np.tile(np.array(color, dtype=np.uint8), (len(positions), 1))

    server.scene.add_batched_meshes_simple(
        name=name,
        vertices=ref_sphere.vertices,
        faces=ref_sphere.faces,
        batched_scales=scale,
        batched_positions=shifted_pos,
        batched_wxyzs=quats,
        batched_colors=colors,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sphere Fitting Comparison Demo")
    parser.add_argument(
        "--robot", default="unitree_g1.yml",
        help="cuRobo robot YAML config name (default: franka.yml)",
    )
    parser.add_argument(
        "--links", nargs="*", default=None,
        help="Link names to visualise (default: auto-select up to 4 links)",
    )
    parser.add_argument(
        "--methods", nargs="*", default=None,
        help="Fit methods to compare (e.g. sample_surface voxel_volume morphit). "
             "Default: all except MORPHIT (add 'morphit' explicitly to include it).",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--num-spheres", type=int, default=0,
                        help="Override sphere count (0 = auto)")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Optimization iterations for iterative methods")
    args = parser.parse_args()

    setup_logger("info")

    # -- Resolve fitting methods to compare -----------------------------------
    if args.methods:
        fit_types = []
        for name in args.methods:
            try:
                fit_types.append(SphereFitType(name))
            except ValueError:
                log_warn(f"Unknown fit type '{name}', skipping")
    else:
        fit_types = [
            SphereFitType.SURFACE,
            SphereFitType.VOXEL,
            SphereFitType.MORPHIT,
        ]

    # -- Load robot -----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Sphere Fitting Comparison Demo")
    print(f"{'=' * 60}\n")

    urdf_parser, mesh_links = _load_parser_from_yml(args.robot)
    print(f"Loaded robot from {args.robot}")
    print(f"  Links with geometry: {len(mesh_links)}")

    # Pick links to visualise
    if args.links:
        selected_links = [ln for ln in args.links if ln in mesh_links]
        if not selected_links:
            log_warn("None of the requested links have geometry, using auto-select")
            selected_links = mesh_links[:4]
    else:
        # Auto-select up to 4 links spread across the chain
        if len(mesh_links) <= 4:
            selected_links = mesh_links
        else:
            indices = np.linspace(0, len(mesh_links) - 1, 4, dtype=int)
            selected_links = [mesh_links[i] for i in indices]

    print(f"  Selected links: {selected_links}")
    print(f"  Fit methods: {[ft.value for ft in fit_types]}")

    # -- Start Viser server ---------------------------------------------------
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.add_grid("/ground_plane", width=4, height=4)
    print(f"\nViser: http://localhost:{args.port}")

    # -- Process each link ----------------------------------------------------
    row_spacing = 0.0  # vertical offset between links (accumulated)

    # Track results with history so the slider can update them.
    # Each entry: (scene_name, color, offset, history)
    history_entries: List[Tuple[str, Tuple[int, int, int], np.ndarray, list]] = []
    max_history_len = 0

    for link_idx, link_name in enumerate(selected_links):
        mesh = _get_link_mesh(urdf_parser, link_name)
        if mesh is None:
            log_warn(f"Skipping {link_name}: no mesh")
            continue

        n_spheres_default = args.num_spheres if args.num_spheres > 0 else estimate_sphere_count(mesh)
        print(f"\n--- {link_name} (default {n_spheres_default} spheres) ---")

        # Compute vertical offset so links don't overlap
        link_height = mesh.bounds[1][2] - mesh.bounds[0][2]
        link_offset_y = row_spacing
        row_spacing += max(link_height, 0.15) + 0.15

        # Add label
        label_pos = np.array([
            -COLUMN_SPACING,
            link_offset_y + link_height / 2,
            (mesh.bounds[0][2] + mesh.bounds[1][2]) / 2,
        ])
        server.scene.add_label(
            f"/labels/{link_name}",
            text=link_name,
            position=label_pos,
        )

        for method_idx, fit_type in enumerate(fit_types):
            col_offset = np.array([
                method_idx * COLUMN_SPACING,
                link_offset_y,
                0.0,
            ])

            num_spheres = n_spheres_default

            result = fit_spheres_to_mesh(
                mesh,
                fit_type=fit_type,
                iterations=args.iterations,
            )

            if result.num_spheres == 0:
                log_warn(f"  {fit_type.value}: no spheres returned")
                continue

            # Show the mesh that was actually used for fitting (may be convex hull)
            vis_mesh = result.used_mesh if result.used_mesh is not None else mesh
            mesh_copy = vis_mesh.copy()
            mesh_copy.visual.vertex_colors = [180, 180, 180, 60]
            server.scene.add_mesh_trimesh(
                f"/links/{link_name}/{fit_type.value}/mesh",
                mesh=mesh_copy,
                position=col_offset,
            )

            print(
                f"  {fit_type.value:30s}  {result.num_spheres:3d} spheres"
                f"  {result.fit_time_s:.2f}s"
            )

            color = METHOD_COLORS.get(fit_type, (200, 200, 200))
            scene_name = f"/links/{link_name}/{fit_type.value}/spheres"
            _add_spheres_to_viser(
                server,
                result.centers.cpu().numpy(),
                result.radii.cpu().numpy(),
                color,
                name=scene_name,
                offset=col_offset,
            )

            if result.history:
                history_entries.append((scene_name, color, col_offset, result.history))
                max_history_len = max(max_history_len, len(result.history))

        # Add method labels at the top of each column (only for first link)
        if link_idx == 0:
            for method_idx, fit_type in enumerate(fit_types):
                label_pos = np.array([
                    method_idx * COLUMN_SPACING,
                    link_offset_y - 0.08,
                    mesh.bounds[1][2] + 0.05,
                ])
                color = METHOD_COLORS.get(fit_type, (200, 200, 200))
                server.scene.add_label(
                    f"/method_labels/{fit_type.value}",
                    text=fit_type.value.replace("_", " "),
                    position=label_pos,
                )

    # -- Add legend via GUI panel ---------------------------------------------
    with server.gui.add_folder("Legend"):
        for ft in fit_types:
            c = METHOD_COLORS.get(ft, (200, 200, 200))
            hex_color = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
            server.gui.add_markdown(
                f'<span style="color:{hex_color}">&#9679;</span> {ft.value.replace("_", " ")}'
            )

    # -- Iteration history slider ---------------------------------------------
    if max_history_len > 1:
        iter_slider = server.gui.add_slider(
            f"Iteration (0–{max_history_len - 1})",
            min=1,
            max=max_history_len,
            step=1,
            initial_value=max_history_len,
        )

        @iter_slider.on_update
        def _on_iter_change(_) -> None:
            idx = int(iter_slider.value) - 1
            for scene_name, color, offset, hist in history_entries:
                frame = min(idx, len(hist) - 1)
                centers, radii = hist[frame]
                _add_spheres_to_viser(
                    server, centers, radii, color, name=scene_name, offset=offset,
                )

    # -- Keep alive -----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Visualisation ready at http://localhost:{args.port}")
    print("Press Ctrl-C to stop.")
    print(f"{'=' * 60}\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
