#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

"""
cuRobo provides accelerated modules for robotics which can be used to build high-performance
robotics applications. The library has several modules for numerical optimization, robot kinematics,
geometry processing, collision checking, graph search planning. cuRobo provides high-level APIs for
performing tasks like collision-free inverse kinematics, model predictive control, and motion
planning.

High-level APIs:

- Motion Generation / Planning: :mod:`curobo.wrap.reacher.motion_gen`.
- Inverse Kinematics: :mod:`curobo.wrap.reacher.ik_solver`.
- Model Predictive Control: :mod:`curobo.wrap.reacher.mpc`.
- Trajectory Optimization: :mod:`curobo.wrap.reacher.trajopt`.


cuRobo package is split into several modules:

- :mod:`curobo.opt` contains optimization solvers.
- :mod:`curobo.cuda_robot_model` contains robot kinematics.
- :mod:`curobo.curobolib` contains the cuda kernels and python bindings for them.
- :mod:`curobo.geom` contains geometry processing, collision checking and frame transforms.
- :mod:`curobo.graph` contains geometric planning with graph search methods.
- :mod:`curobo.rollout` contains methods that map actions to costs. This class wraps instances of
  :mod:`curobo.cuda_robot_model` and :mod:`curobo.geom` to compute costs given trajectory of actions.
- :mod:`curobo.util` contains utility methods.
- :mod:`curobo.wrap` adds the user-level api for task programming. Includes implementation of
  collision-free reacher and batched robot world collision checking.
- :mod:`curobo.types` contains custom dataclasses for common data types in robotics, including
  :py:meth:`~types.state.JointState`, :py:meth:`~types.camera.CameraObservation`,
  :py:meth:`~types.math.Pose`.
"""


# NOTE (roflaherty): This is inspired by how matplotlib does creates its version value.
# https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/__init__.py#L161
def _get_version():
    """Return the version string used for __version__."""
    # Standard Library
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent.parent
    if (root / ".git").exists() and not (root / ".git/shallow").exists():
        # Third Party
        import setuptools_scm

        # See the `setuptools_scm` documentation for the description of the schemes used below.
        # https://pypi.org/project/setuptools-scm/
        # NOTE: If these values are updated, they need to be also updated in `pyproject.toml`.
        return setuptools_scm.get_version(
            root=root,
            version_scheme="no-guess-dev",
            local_scheme="dirty-tag",
        )
    else:  # Get the version from the _version.py setuptools_scm file.
        try:
            # Standard Library
            from importlib.metadata import version
        except ModuleNotFoundError:
            # NOTE: `importlib.resources` is part of the standard library in Python 3.9.
            # `importlib_metadata` is the back ported library for older versions of python.
            # Third Party
            from importlib_metadata import version
        try:
            return version("nvidia_curobo")
        except:
            return "v0.7.0-no-tag"


# Set `__version__` attribute
__version__ = _get_version()

# Remove `_get_version` so it is not added as an attribute
del _get_version
