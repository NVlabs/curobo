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
# Standard Library
import os
import shutil
from typing import Dict, List

# Third Party
import yaml
from yaml import Loader


# get paths
def get_module_path():
    path = os.path.dirname(__file__)
    return path


def get_root_path():
    path = os.path.dirname(get_module_path())
    return path


def get_content_path():
    root_path = get_module_path()
    path = os.path.join(root_path, "content")
    return path


def get_configs_path():
    content_path = get_content_path()
    path = os.path.join(content_path, "configs")
    return path


def get_assets_path():
    content_path = get_content_path()
    path = os.path.join(content_path, "assets")
    return path


def get_weights_path():
    content_path = get_content_path()
    path = os.path.join(content_path, "weights")
    return path


def join_path(path1, path2):
    if isinstance(path2, str):
        return os.path.join(path1, path2)
    else:
        return path2


def load_yaml(file_path):
    if isinstance(file_path, str):
        with open(file_path) as file_p:
            yaml_params = yaml.load(file_p, Loader=Loader)
    else:
        yaml_params = file_path
    return yaml_params


def write_yaml(data: Dict, file_path):
    with open(file_path, "w") as file:
        yaml.dump(data, file)


def get_robot_path():
    config_path = get_configs_path()
    path = os.path.join(config_path, "robot")
    return path


def get_task_configs_path():
    config_path = get_configs_path()
    path = os.path.join(config_path, "task")
    return path


def get_robot_configs_path():
    config_path = get_configs_path()
    path = os.path.join(config_path, "robot")
    return path


def get_world_configs_path():
    config_path = get_configs_path()
    path = os.path.join(config_path, "world")
    return path


def get_debug_path():
    asset_path = get_assets_path()
    path = join_path(asset_path, "debug")
    return path


def get_cpp_path():
    path = os.path.dirname(__file__)
    return os.path.join(path, "curobolib/cpp")


def add_cpp_path(sources):
    cpp_path = get_cpp_path()
    new_list = []
    for s in sources:
        s = join_path(cpp_path, s)
        new_list.append(s)
    return new_list


def copy_file_to_path(source_file, destination_path):
    #
    isExist = os.path.exists(destination_path)
    if not isExist:
        os.makedirs(destination_path)
    _, file_name = os.path.split(source_file)
    new_path = join_path(destination_path, file_name)
    isExist = os.path.exists(new_path)
    if not isExist:
        shutil.copyfile(source_file, new_path)
    return new_path


def get_filename(file_path, remove_extension: bool = False):
    _, file_name = os.path.split(file_path)
    if remove_extension:
        file_name = os.path.splitext(file_name)[0]
    return file_name


def get_path_of_dir(file_path):
    dir_path, _ = os.path.split(file_path)
    return dir_path


def get_files_from_dir(dir_path, extension: List[str], contains: str):
    file_names = [
        fn
        for fn in os.listdir(dir_path)
        if (any(fn.endswith(ext) for ext in extension) and contains in fn)
    ]
    file_names.sort()
    return file_names


def file_exists(path):
    isExist = os.path.exists(path)
    return isExist


def get_motion_gen_robot_list() -> List[str]:
    """returns list of robots available in curobo for motion generation

    Returns:
        _description_
    """
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
    return get_motion_gen_robot_list()


def get_multi_arm_robot_list() -> List[str]:
    robot_list = [
        "dual_ur10e.yml",
        "tri_ur10e.yml",
        "quad_ur10e.yml",
    ]
    return robot_list
