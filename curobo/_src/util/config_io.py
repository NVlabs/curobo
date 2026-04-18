# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Internal file utilities (private)."""

# Standard Library
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, TypeVar, Union

# Third Party
import yaml
from yaml import CLoader as Loader

# CuRobo

# YAML loader configuration for proper float parsing
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


def join_path(path1: Union[str, Path], path2: Union[str, Path]) -> str:
    """Join two paths, considering OS specific path separators.

    Args:
        path1: Path prefix.
        path2: Path suffix. If path2 is an absolute path, path1 is ignored.

    Returns:
        str: Joined path as per standard path joining semantics:
        - If path2 is absolute, return path2 (ignore path1)
        - If path2 is relative, join path1 and path2
    """
    if isinstance(path1, Path):
        path1 = str(path1)
    if isinstance(path2, Path):
        path2 = str(path2)
    if not isinstance(path2, str):
        return path2

    # NOTE: verify is returned path is absolute with a leading slash
    # Standard path joining: if path2 is absolute, return it as-is
    return os.path.join(path1, path2)


ConfigT = TypeVar("ConfigT")


def resolve_config(config: Union[str, ConfigT]) -> Union[Dict, ConfigT]:
    """Resolve configuration from file path or return as-is.

    If config is a string (file path), loads and returns the YAML content as a dict.
    Otherwise, returns config unchanged. This allows passing:
    - A file path (str) to load from YAML
    - A dict (already parsed config)
    - A typed config object (e.g., RobotCfg) that passes through unchanged

    Args:
        config: A path to a YAML file, a dict, or an already-constructed config object.

    Returns:
        If config is a string, loads and returns the YAML as a dict.
        Otherwise, returns config unchanged.
    """
    if isinstance(config, str):
        with open(config) as file_p:
            return yaml.load(file_p, Loader=Loader)
    return config


def load_yaml(file_path: Union[str, Dict]) -> Dict:
    """Load yaml file and return as dictionary. If file_path is a dictionary, return as is.

    Note:
        This function calls :func:`resolve_config` internally. Consider using
        :func:`resolve_config` directly for more flexibility with typed config objects.

    Args:
        file_path: File path to yaml file or dictionary.

    Returns:
        Dict: Dictionary containing yaml file content.
    """
    return resolve_config(file_path)


def write_yaml(data: Dict, file_path: str):
    """Write dictionary to yaml file.

    Args:
        data: Dictionary to write to yaml file.
        file_path: Path to write the yaml file.
    """
    with open(file_path, "w") as file:
        yaml.dump(data, file)


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


def create_dir_if_not_exists(dir_path: str):
    """Create directory if it does not exist.

    Args:
        dir_path: Path of directory.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
