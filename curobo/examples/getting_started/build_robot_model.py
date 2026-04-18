# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Build a cuRobo robot configuration from a URDF file.

cuRobo needs two pieces of information that a standard URDF does not contain:
**collision spheres** fitted to each link mesh and a **self-collision ignore
matrix** that lists link pairs whose collisions can be safely skipped. The
:class:`~curobo.robot_builder.RobotBuilder` API generates both and writes them
to a YAML or XRDF config file consumed by every downstream cuRobo module
(forward kinematics, inverse kinematics, motion planning, MPC).

.. image:: /images/franka_spheres.png
   :width: 100%
   :alt: Robot mesh (left) and its collision sphere approximation (right)

By the end of this tutorial you will have:

- Created a cuRobo robot configuration from a URDF file
- Fitted collision spheres to each link mesh using the MorphIt optimizer
- Computed an optimized self-collision ignore matrix
- Inspected per-link sphere fit quality metrics
- Visualized the fitted spheres in a browser
- Saved the configuration in both YAML and XRDF formats

Step 1: Prepare your URDF
--------------------------

You need a URDF file and a directory containing the referenced mesh assets.
cuRobo ships robot URDFs under ``curobo/content/assets/robot/``, so the
tutorial works out of the box with the bundled Franka Panda. For your own
robot, ``package://`` prefixes in mesh filenames are stripped automatically;
set ``--asset-path`` to the directory that contains the remaining relative
path (e.g., if the URDF says ``package://my_robot/meshes/link.stl``, point
``--asset-path`` at the parent of ``my_robot/``).

If your robot needs a **floating base** (e.g., a humanoid whose pelvis moves
freely in space), use ``extra_links`` with ``child_link_name`` in the YAML config
to insert virtual joints between ``base_link`` and the robot's root body. This
requires no URDF modification. See the
:ref:`humanoid retargeting guide <humanoid_retargeting>` for a
complete example using the Unitree G1.

Step 2: Run the tutorial
--------------------------

.. code-block:: bash

   python -m curobo.examples.getting_started.build_robot_model \\
       --urdf curobo/content/assets/robot/franka_description/franka_panda.urdf \\
       --asset-path curobo/content/assets/robot/franka_description \\
       --output franka_custom.yml \\
       --clip-link panda_link0 z 0.0 \\
       --compute-metrics

Add ``--visualize`` to inspect the fitted spheres in a
`Viser <https://viser.studio>`_ viewer at ``http://localhost:8080``.
``--clip-link panda_link0 z 0.0`` keeps the base link's spheres from
protruding below the mounting surface (see :ref:`Step 5 <step5_tuning>`).

Step 3: Check the output
--------------------------

When the tutorial finishes successfully you will see::

    Building robot model from URDF: ...franka_panda.urdf
    Found 11 links in robot

    Fitting collision spheres...
    Fitted 87 spheres across 9 links

      link                      n_sph  cover%  protr%   prot_mm  gap_mm  vol_ratio
      ---------------------------------------------------------------------------
      panda_link0                  12   98.3%    4.1%     0.72mm   1.04mm     1.124
      ...

    Computing collision matrix...
    Created collision ignore matrix with 28 entries

    Saving to: franka_custom.yml
    ✓ Robot model created successfully!

The generated ``franka_custom.yml`` (or ``.xrdf`` if you chose that extension)
is ready to use with :mod:`~curobo.wrap.reacher.ik_solver`,
:mod:`~curobo.wrap.reacher.motion_gen`, or any other cuRobo wrapper.

Step 4: Understand the pipeline
---------------------------------

The builder runs three stages:

1. **Sphere fitting**: Each link mesh is approximated by a set of spheres
   using the MorphIt optimizer (an Adam-based iterative fit that balances
   interior *coverage* against surface *protrusion*). The ``--sphere-density``
   multiplier controls how many spheres are allocated per link.

2. **Self-collision matrix**: Link pairs that are always in collision (e.g.
   adjacent joints) or never reachable are identified via random joint
   sampling (``--num-collision-samples``) and placed in an ignore set so the
   downstream planner skips them.

3. **Export**: The sphere and matrix data, together with the kinematic
   tree, are serialized to YAML (native cuRobo) or XRDF (Isaac Sim / Isaac
   Lab). Use ``--export-xrdf`` to emit both formats at once.

Step 5: Tuning (advanced)
---------------------------

- ``--coverage-weight`` / ``--protrusion-weight`` control the MorphIt loss
  balance. Raise ``coverage-weight`` (default 1000) for tighter volume
  filling; raise ``protrusion-weight`` (default 10) to reduce overshoot.
- ``--sphere-density 2.0`` doubles the sphere budget per link.
- ``--edit-config existing.yml --refit-link panda_hand`` re-fits spheres for
  a single link without re-running the entire pipeline.
- ``--seed 42`` pins NumPy and PyTorch RNGs for reproducible results.
- ``--clip-link base_link z 0.0`` prevents spheres on ``base_link`` from
  extending below ``z=0`` in link-local coordinates.  This is useful when the
  robot is mounted on a stand or bolted to the floor -- without clipping, the
  base link spheres may overlap the mounting surface and cause perpetual
  collisions.  The constraint is enforced as both a differentiable MorphIt
  loss and a hard post-fit clamp.  Can be repeated for multiple links.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from curobo.content import get_assets_path
from curobo.logging import setup_logger
from curobo.robot_builder import RobotBuilder


def build_new_robot(args):
    """Build a new robot model from URDF."""
    print(f"Building robot model from URDF: {args.urdf}")
    print(f"Asset path: {args.asset_path}")

    # Create builder
    builder = RobotBuilder(
        urdf_path=args.urdf,
        asset_path=args.asset_path,
        tool_frames=args.tool_frames,
    )

    print(f"Found {len(builder.tool_frames)} links in robot")

    clip_links = None
    if args.clip_link:
        clip_links = {link: (axis, float(offset)) for link, axis, offset in args.clip_link}

    # Fit collision spheres
    print("\nFitting collision spheres...")
    builder.fit_collision_spheres(
        sphere_density=args.sphere_density,
        coverage_weight=args.coverage_weight,
        protrusion_weight=args.protrusion_weight,
        compute_metrics=args.compute_metrics,
        clip_links=clip_links,
    )

    print(f"Fitted {builder.num_spheres} spheres across {len(builder.collision_link_names)} links")

    if args.compute_metrics and builder.link_metrics:
        header = (
            f"  {'link':<25s} {'n_sph':>5s} "
            f"{'cover%':>7s} {'protr%':>7s} {'prot_mm':>8s} "
            f"{'gap_mm':>7s} {'vol_ratio':>9s}"
        )
        print(f"\n{header}")
        print(f"  {'-' * (len(header) - 2)}")
        metrics_list = list(builder.link_metrics.values())
        for link_name, m in builder.link_metrics.items():
            print(
                f"  {link_name:<25s} {m.num_spheres:5d} "
                f"{m.coverage * 100:6.1f}% {m.protrusion * 100:6.1f}% "
                f"{m.protrusion_dist_mean * 1000:7.2f}mm "
                f"{m.surface_gap_mean * 1000:6.2f}mm "
                f"{m.volume_ratio:9.3f}"
            )
        n = len(metrics_list)
        print(f"  {'-' * (len(header) - 2)}")
        total_spheres = sum(m.num_spheres for m in metrics_list)
        print(
            f"  {'TOTAL / AVG':<25s} {total_spheres:5d} "
            f"{sum(m.coverage for m in metrics_list) / n * 100:6.1f}% "
            f"{sum(m.protrusion for m in metrics_list) / n * 100:6.1f}% "
            f"{sum(m.protrusion_dist_mean for m in metrics_list) / n * 1000:7.2f}mm "
            f"{sum(m.surface_gap_mean for m in metrics_list) / n * 1000:6.2f}mm "
            f"{sum(m.volume_ratio for m in metrics_list) / n:9.3f}"
        )

    # Compute collision matrix
    print("\nComputing collision matrix...")
    builder.compute_collision_matrix(
        prune_collisions=not args.no_prune,
        num_samples=args.num_collision_samples,
    )

    print(f"Created collision ignore matrix with {len(builder.collision_matrix)} entries")

    # Build configuration
    print("\nBuilding configuration...")
    config = builder.build()

    # Save
    print(f"Saving to: {args.output}")

    # Determine format from file extension
    if args.output.endswith('.xrdf'):
        builder.save_xrdf(config, args.output)
    else:
        builder.save(config, args.output)

    # Also save in alternate format if requested
    if args.export_xrdf and not args.output.endswith('.xrdf'):
        xrdf_path = args.output.replace('.yml', '.xrdf').replace('.yaml', '.xrdf')
        print(f"Also exporting to XRDF: {xrdf_path}")
        builder.save_xrdf(config, xrdf_path)

    print("\n✓ Robot model created successfully!")

    # Visualize if requested
    if args.visualize:
        print(f"\nStarting visualization server at http://localhost:{args.viz_port}")
        print("Press Ctrl+C to stop")
        viser = builder.visualize(config, port=args.viz_port)
        try:
            import time

            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping visualization")


def edit_existing_robot(args):
    """Edit an existing robot configuration."""
    print(f"Loading robot configuration: {args.edit_config}")

    # Load existing config
    builder = RobotBuilder.from_config(args.edit_config)

    print(f"Loaded robot with {builder.num_spheres} spheres")

    # Refit specific link if requested
    if args.refit_link:
        print(f"\nRefitting spheres for link: {args.refit_link}")
        new_spheres = builder.refit_link_spheres(
            args.refit_link,
            sphere_density=args.sphere_density,
        )
        print(f"Fitted {len(new_spheres)} spheres to {args.refit_link}")

    # Add collision ignores if requested
    if args.add_collision_ignore:
        link_name, ignore_links = args.add_collision_ignore
        ignore_list = ignore_links.split(",")
        print(f"\nAdding collision ignores: {link_name} -> {ignore_list}")
        builder.add_collision_ignore(link_name, ignore_list)

    # Recompute collision matrix if requested
    if args.recompute_collisions:
        print("\nRecomputing collision matrix...")
        builder.compute_collision_matrix(
            prune_collisions=not args.no_prune,
            num_samples=args.num_collision_samples,
        )

    # Build and save
    print("\nBuilding updated configuration...")
    config = builder.build()

    print(f"Saving to: {args.output}")

    # Determine format from file extension
    if args.output.endswith('.xrdf'):
        builder.save_xrdf(config, args.output)
    else:
        builder.save(config, args.output)

    # Also save in alternate format if requested
    if args.export_xrdf and not args.output.endswith('.xrdf'):
        xrdf_path = args.output.replace('.yml', '.xrdf').replace('.yaml', '.xrdf')
        print(f"Also exporting to XRDF: {xrdf_path}")
        builder.save_xrdf(config, xrdf_path)

    print("\n✓ Robot model updated successfully!")

    # Visualize if requested
    if args.visualize:
        print(f"\nStarting visualization server at http://localhost:{args.viz_port}")
        print("Press Ctrl+C to stop")
        viser = builder.visualize(config, port=args.viz_port)
        try:
            import time

            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping visualization")


def test():
    """Run build_robot_model with the bundled Franka URDF as a self-test.

    Tests three modes:
    1. Build new robot from URDF
    2. Edit existing config: refit a single link
    3. Edit existing config: add collision ignore + recompute collisions
    """
    assets = get_assets_path()
    urdf = str(assets / "robot" / "franka_description" / "franka_panda.urdf")
    asset_path = str(assets / "robot" / "franka_description")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Build new robot
        output = str(Path(tmpdir) / "franka_test.yml")
        build_args = argparse.Namespace(
            urdf=urdf,
            edit_config=None,
            asset_path=asset_path,
            output=output,
            export_xrdf=False,
            tool_frames=[],
            sphere_density=1.0,
            coverage_weight=None,
            protrusion_weight=None,
            compute_metrics=True,
            clip_link=None,
            num_collision_samples=100,
            no_prune=False,
            refit_link=None,
            add_collision_ignore=None,
            recompute_collisions=False,
            visualize=False,
            viz_port=8080,
            seed=42,
            log_level="warning",
        )
        np.random.seed(42)
        torch.manual_seed(42)
        build_new_robot(build_args)
        assert Path(output).exists(), f"Output file not created: {output}"

        # 2. Edit: refit a single link
        refit_output = str(Path(tmpdir) / "franka_refit.yml")
        builder = RobotBuilder.from_config(output)
        refit_link = builder.collision_link_names[0]

        refit_args = argparse.Namespace(
            urdf=None,
            edit_config=output,
            asset_path="",
            output=refit_output,
            export_xrdf=False,
            tool_frames=[],
            sphere_density=1.0,
            coverage_weight=None,
            protrusion_weight=None,
            compute_metrics=False,
            clip_link=None,
            num_collision_samples=100,
            no_prune=False,
            refit_link=refit_link,
            add_collision_ignore=None,
            recompute_collisions=False,
            visualize=False,
            viz_port=8080,
            seed=42,
            log_level="warning",
        )
        edit_existing_robot(refit_args)
        assert Path(refit_output).exists(), f"Refit output not created: {refit_output}"

        # 3. Edit: add collision ignore + recompute collisions
        ignore_output = str(Path(tmpdir) / "franka_ignore.yml")
        link_names = builder.collision_link_names
        ignore_args = argparse.Namespace(
            urdf=None,
            edit_config=output,
            asset_path="",
            output=ignore_output,
            export_xrdf=False,
            tool_frames=[],
            sphere_density=1.0,
            coverage_weight=None,
            protrusion_weight=None,
            compute_metrics=False,
            clip_link=None,
            num_collision_samples=100,
            no_prune=False,
            refit_link=None,
            add_collision_ignore=[link_names[0], link_names[1]] if len(link_names) > 1 else None,
            recompute_collisions=True,
            visualize=False,
            viz_port=8080,
            seed=42,
            log_level="warning",
        )
        edit_existing_robot(ignore_args)
        assert Path(ignore_output).exists(), f"Ignore output not created: {ignore_output}"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build robot model configuration from URDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--urdf",
        type=str,
        help="Path to URDF file (for creating new robot model)",
    )
    mode_group.add_argument(
        "--edit-config",
        type=str,
        help="Path to existing .yml config (for editing)",
    )
    mode_group.add_argument(
        "--test",
        action="store_true",
        help="Run as self-test using bundled Franka URDF",
    )

    # Common arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output file (.yml or .xrdf)",
    )
    parser.add_argument(
        "--export-xrdf",
        action="store_true",
        help="Also export to XRDF format (if output is .yml)",
    )

    # New robot arguments
    parser.add_argument(
        "--asset-path",
        type=str,
        default="",
        help="Path to mesh assets (required when using --urdf)",
    )
    parser.add_argument(
        "--tool-frames",
        nargs="+",
        type=str,
        default=[],
        help="Tool frames (optional)",
    )

    # Sphere fitting arguments
    parser.add_argument(
        "--sphere-density",
        type=float,
        default=1.0,
        help="Sphere density multiplier (default: 1.0; higher = more spheres)",
    )
    parser.add_argument(
        "--coverage-weight",
        type=float,
        default=None,
        help="MorphIt coverage loss weight (default: 1000.0; higher = better volume filling)",
    )
    parser.add_argument(
        "--protrusion-weight",
        type=float,
        default=None,
        help="MorphIt protrusion loss weight (default: 10.0; higher = less overshoot)",
    )
    parser.add_argument(
        "--compute-metrics",
        action="store_true",
        help="Compute and print per-link sphere fit quality metrics",
    )
    parser.add_argument(
        "--clip-link",
        nargs=3,
        action="append",
        metavar=("LINK", "AXIS", "OFFSET"),
        help=(
            "Clip spheres on LINK so they don't extend past a plane. "
            "AXIS is x/y/z (prefix with - for negative direction), "
            "OFFSET is the plane position. Can be repeated. "
            "Example: --clip-link base_link z 0.0"
        ),
    )

    # Collision matrix arguments
    parser.add_argument(
        "--num-collision-samples",
        type=int,
        default=1000,
        help="Number of samples for collision pruning (default: 1000)",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Skip collision pruning (faster but less optimized)",
    )

    # Edit mode arguments
    parser.add_argument(
        "--refit-link",
        type=str,
        help="Refit spheres for specific link (edit mode)",
    )
    parser.add_argument(
        "--add-collision-ignore",
        nargs=2,
        metavar=("LINK", "IGNORE_LINKS"),
        help="Add collision ignore: LINK IGNORE_LINKS (comma-separated)",
    )
    parser.add_argument(
        "--recompute-collisions",
        action="store_true",
        help="Recompute entire collision matrix (edit mode)",
    )

    # Visualization arguments
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Start visualization server after building",
    )
    parser.add_argument(
        "--viz-port",
        type=int,
        default=8080,
        help="Visualization server port (default: 8080)",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic sphere fitting",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="warning",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)",
    )

    args = parser.parse_args()

    if args.test:
        test()
        sys.exit(0)

    if args.output is None:
        parser.error("--output is required when using --urdf or --edit-config")

    setup_logger(args.log_level)

    # Seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Validate arguments
    if args.urdf and not args.asset_path:
        parser.error("--asset-path is required when using --urdf")

    if args.edit_config and any(
        [args.urdf, args.asset_path, args.tool_frames]
    ):
        parser.error("Cannot use --urdf, --asset-path, or --tool-frames with --edit-config")

    # Run appropriate mode
    try:
        if args.urdf:
            build_new_robot(args)
        else:
            edit_existing_robot(args)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        if args.log_level == "debug":
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()

