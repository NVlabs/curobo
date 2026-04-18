# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visualization module.

This module provides visualization backends for CuRobo. Viser and ``usd-core`` are optional
dependencies and are imported lazily on first use.

Available visualizers:
    - :func:`ViserVisualizer`: Interactive 3D visualization with a web interface.
    - :func:`UsdWriter`: Offline export of robot trajectories to USD for OpenUSD-compatible
      viewers (e.g., Isaac Sim, usdview).

Example (Viser):
    ```python
    from curobo.viewer import ViserVisualizer
    from curobo.types import JointState

    viz = ViserVisualizer(
        robot_config="franka.yml",
        connect_ip="0.0.0.0",
        connect_port=8080,
        add_control_frames=True,
    )
    # Opens web interface at http://localhost:8080
    joint_state = JointState.from_position([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
    viz.set_joint_state(joint_state)
    ```

Example (USD):
    ```python
    from curobo.viewer import UsdWriter

    writer = UsdWriter("trajectory.usd", robot_config="franka.yml")
    writer.write(joint_trajectory)
    ```
"""


def ViserVisualizer(*args, **kwargs):
    """Create Viser visualizer for interactive 3D visualization.

    Viser provides an interactive web-based 3D viewer with real-time
    control capabilities, including draggable control frames.

    Args:
        *args: Positional arguments passed to ViserVisualizer
        **kwargs: Keyword arguments passed to ViserVisualizer

    Returns:
        ViserVisualizer instance

    Raises:
        ImportError: If viser is not installed.
            Install with: pip install viser
    """
    try:
        from curobo._src.util.viser_visualizer import ViserVisualizer as ViserVisualizerImpl

        return ViserVisualizerImpl(*args, **kwargs)
    except ImportError as e:
        raise ImportError("Viser not installed. Install with: pip install viser") from e


def UsdWriter(*args, **kwargs):
    """Create a USD writer for exporting robot trajectories to OpenUSD files.

    ``UsdWriter`` serializes robot geometry and joint trajectories to a USD stage that can
    be inspected in any OpenUSD-compatible viewer (Isaac Sim, usdview, etc.).

    Args:
        *args: Positional arguments passed to the underlying USD writer.
        **kwargs: Keyword arguments passed to the underlying USD writer.

    Returns:
        USD writer instance.

    Raises:
        ImportError: If ``usd-core`` is not installed.
            Install with: ``pip install usd-core`` (skip this when running inside Isaac Sim).
    """
    try:
        from curobo._src.util.usd_writer import UsdWriter as UsdWriterImpl
    except ImportError as e:
        raise ImportError(
            "usd-core not installed. Install with: pip install usd-core "
            "(skip this when running inside Isaac Sim)"
        ) from e
    return UsdWriterImpl(*args, **kwargs)


__all__ = ["ViserVisualizer", "UsdWriter"]
