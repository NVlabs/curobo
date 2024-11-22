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
"""Contains helper functions for interacting with file systems."""
# Standard Library
import os
import re
import shutil
import sys
from typing import Any, Dict, List, Union

# Third Party
import yaml
from yaml import SafeLoader as Loader

# CuRobo
from curobo.util.logger import log_warn

Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
    [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


# get paths
def get_module_path() -> str:
    """Get absolute path of cuRobo library."""
    path = os.path.dirname(__file__)
    return path


def get_root_path() -> str:
    """Get absolute path of cuRobo library."""
    path = os.path.dirname(get_module_path())
    return path


def get_content_path() -> str:
    """Get path to content directory in cuRobo.

    Content directory contains configuration parameters for different tasks, some robot
    parameters for using in examples, and some world assets. Use
    :class:`~curobo.util.file_path.ContentPath` when running cuRobo with assets from a different
    location.

    Returns:
        str: path to content directory.
    """
    root_path = get_module_path()
    path = os.path.join(root_path, "content")
    return path


def get_configs_path() -> str:
    """Get path to configuration parameters for different tasks(e.g., IK, TrajOpt, MPC) in cuRobo.

    Returns:
        str: path to configuration directory.
    """
    content_path = get_content_path()
    path = os.path.join(content_path, "configs")
    return path


def get_assets_path() -> str:
    """Get path to assets (robot urdf, meshes, world meshes) directory in cuRobo."""

    content_path = get_content_path()
    path = os.path.join(content_path, "assets")
    return path


def get_weights_path():
    """Get path to neural network weights directory in cuRobo. Currently not used in cuRobo."""
    content_path = get_content_path()
    path = os.path.join(content_path, "weights")
    return path


def join_path(path1: str, path2: str) -> str:
    """Join two paths, considering OS specific path separators.

    Args:
        path1: Path prefix.
        path2: Path suffix. If path2 is an absolute path, path1 is ignored.

    Returns:
        str: Joined path.
    """
    if path1[-1] == os.sep:
        log_warn("path1 has trailing slash, removing it")
    if isinstance(path2, str):
        return os.path.join(os.sep, path1 + os.sep, path2)
    else:
        return path2


def load_yaml(file_path: Union[str, Dict]) -> Dict:
    """Load yaml file and return as dictionary. If file_path is a dictionary, return as is.

    Args:
        file_path: File path to yaml file or dictionary.

    Returns:
        Dict: Dictionary containing yaml file content.
    """
    if isinstance(file_path, str):
        with open(file_path) as file_p:
            yaml_params = yaml.load(file_p, Loader=Loader)
    else:
        yaml_params = file_path
    return yaml_params


def write_yaml(data: Dict, file_path: str):
    """Write dictionary to yaml file.

    Args:
        data: Dictionary to write to yaml file.
        file_path: Path to write the yaml file.
    """
    with open(file_path, "w") as file:
        yaml.dump(data, file)


def get_robot_path() -> str:
    """Get path to robot directory in cuRobo.

    Deprecated: Use :func:`~curobo.util_file.get_robot_configs_path` instead.
    Robot directory contains robot configuration files in yaml format. See
    :ref:`tut_robot_configuration` for how to create a robot configuration file.

    Returns:
        str: path to robot directory.
    """
    config_path = get_configs_path()
    path = os.path.join(config_path, "robot")
    return path


def get_task_configs_path() -> str:
    """Get path to task configuration directory in cuRobo.

    Task directory contains configuration parameters for different tasks (e.g., IK, TrajOpt, MPC).

    Returns:
        str: path to task configuration directory.
    """
    config_path = get_configs_path()
    path = os.path.join(config_path, "task")
    return path


def get_robot_configs_path() -> str:
    """Get path to robot configuration directory in cuRobo.

    Robot configuration directory contains robot configuration files in yaml format. See
    :ref:`tut_robot_configuration` for how to create a robot configuration file.

    Returns:
        str: path to robot configuration directory.
    """
    config_path = get_configs_path()
    path = os.path.join(config_path, "robot")
    return path


def get_world_configs_path() -> str:
    """Get path to world configuration directory in cuRobo.

    World configuration directory contains world configuration files in yaml format. World
    information includes obstacles represented with respect to the robot base frame.

    Returns:
        str: path to world configuration directory.
    """
    config_path = get_configs_path()
    path = os.path.join(config_path, "world")
    return path


def get_debug_path() -> str:
    """Get path to debug directory in cuRobo.

    Debug directory can be used to store logs and debug information.

    Returns:
        str: path to debug directory.
    """

    asset_path = get_assets_path()
    path = join_path(asset_path, "debug")
    return path


def get_cpp_path():
    """Get path to cpp directory in cuRobo.

    Directory contains CUDA implementations (kernels) of robotics algorithms, which are wrapped
    in C++ and compiled with PyTorch to enable usage in Python.

    Returns:
        str: path to cpp directory.
    """
    path = os.path.dirname(__file__)
    return os.path.join(path, "curobolib/cpp")


def add_cpp_path(sources: List[str]) -> List[str]:
    """Add cpp path to list of source files.

    Args:
        sources: List of source files.

    Returns:
        List[str]: List of source files with cpp path added.
    """
    cpp_path = get_cpp_path()
    new_list = []
    for s in sources:
        s = join_path(cpp_path, s)
        new_list.append(s)
    return new_list


def copy_file_to_path(source_file: str, destination_path: str) -> str:
    """Copy file from source to destination.

    Args:
        source_file: Path of source file.
        destination_path: Path of destination directory.

    Returns:
        str: Destination path of copied file.
    """
    isExist = os.path.exists(destination_path)
    if not isExist:
        os.makedirs(destination_path)
    _, file_name = os.path.split(source_file)
    new_path = join_path(destination_path, file_name)
    isExist = os.path.exists(new_path)
    if not isExist:
        shutil.copyfile(source_file, new_path)
    return new_path


def get_filename(file_path: str, remove_extension: bool = False) -> str:
    """Get file name from file path, removing extension if required.

    Args:
        file_path: Path of file.
        remove_extension: If True, remove file extension.

    Returns:
        str: File name.
    """

    _, file_name = os.path.split(file_path)
    if remove_extension:
        file_name = os.path.splitext(file_name)[0]
    return file_name


def get_path_of_dir(file_path: str) -> str:
    """Get path of directory containing the file.

    Args:
        file_path: Path of file.

    Returns:
        str: Path of directory containing the file.
    """
    dir_path, _ = os.path.split(file_path)
    return dir_path


def get_files_from_dir(dir_path, extension: List[str], contains: str) -> List[str]:
    """Get list of files from directory with specified extension and containing a string.

    Args:
        dir_path: Path of directory.
        extension: List of file extensions to filter.
        contains: String to filter file names.

    Returns:
        List[str]: List of file names. Does not include path.
    """
    file_names = [
        fn
        for fn in os.listdir(dir_path)
        if (any(fn.endswith(ext) for ext in extension) and contains in fn)
    ]
    file_names.sort()
    return file_names


def file_exists(path: str) -> bool:
    """Check if file exists.

    Args:
        path: Path of file.

    Returns:
        bool: True if file exists, False otherwise.
    """
    if path is None:
        return False
    isExist = os.path.exists(path)
    return isExist


def get_motion_gen_robot_list() -> List[str]:
    """Get list of robot configuration examples in cuRobo for motion generation."""
    robot_list = [
        "franka.yml",
        "ur5e.yml",
        "ur10e.yml",
        "tm12.yml",
        "jaco7.yml",
        "kinova_gen3.yml",
        "iiwa.yml",
        "iiwa_allegro.yml",
        # "franka_mobile.yml",
    ]
    return robot_list


def get_robot_list() -> List[str]:
    """Get list of robots example configurations in cuRobo."""
    return get_motion_gen_robot_list()


def get_multi_arm_robot_list() -> List[str]:
    """Get list of multi-arm robot configuration examples in cuRobo."""
    robot_list = [
        "dual_ur10e.yml",
        "tri_ur10e.yml",
        "quad_ur10e.yml",
    ]
    return robot_list


def merge_dict_a_into_b(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Merge dictionary values in "a" into dictionary "b". Overwrite values in "b" if key exists.

    Args:
        a: New dictionary to merge.
        b: Base dictionary to merge into.

    Returns:
        Merged dictionary.
    """
    for k, v in a.items():
        if isinstance(v, dict):
            merge_dict_a_into_b(v, b[k])
        else:
            b[k] = v
    return b


def is_platform_windows() -> bool:
    """Check if platform is Windows."""
    return sys.platform == "win32"


def is_platform_linux() -> bool:
    """Check if platform is Linux."""
    return sys.platform == "linux"


def is_file_xrdf(file_path: str) -> bool:
    """Check if file is an `XRDF <https://nvidia-isaac-ros.github.io/concepts/manipulation/xrdf.html>`_ file.

    Args:
        file_path: Path of file.

    Returns:
        bool: True if file is xrdf, False otherwise.
    """
    if file_path.endswith(".xrdf") or file_path.endswith(".XRDF"):
        return True
    return False
