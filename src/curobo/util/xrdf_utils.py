# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Standard Library
from copy import deepcopy
from typing import Any, Dict, Optional

# CuRobo
from curobo.cuda_robot_model.urdf_kinematics_parser import UrdfKinematicsParser
from curobo.types.file_path import ContentPath
from curobo.util.logger import log_error, log_warn
from curobo.util_file import load_yaml


def return_value_if_exists(
    input_dict: Dict, key: str, suffix: str = "xrdf", raise_error: bool = True
) -> Any:
    if key not in input_dict:
        if raise_error:
            log_error(key + " key not found in " + suffix)
        return None
    return input_dict[key]


def convert_xrdf_to_curobo(
    content_path: ContentPath = ContentPath(),
    input_xrdf_dict: Optional[Dict] = None,
) -> Dict:

    if content_path.robot_urdf_absolute_path is None:
        log_error(
            "content_path.robot_urdf_absolute_path or content_path.robot_urdf_file \
                 is required."
        )
    urdf_path = content_path.robot_urdf_absolute_path
    if input_xrdf_dict is None:
        input_xrdf_dict = load_yaml(content_path.robot_xrdf_absolute_path)

    if isinstance(content_path, str):
        log_error("content_path should be of type ContentPath")

    if return_value_if_exists(input_xrdf_dict, "format") != "xrdf":
        log_error("format is not xrdf")

    if return_value_if_exists(input_xrdf_dict, "format_version") > 1.0:
        log_warn("format_version is greater than 1.0")
    # Also get base link as root of urdf
    kinematics_parser = UrdfKinematicsParser(
        urdf_path, mesh_root=content_path.robot_asset_absolute_path, build_scene_graph=True
    )
    joint_names = kinematics_parser.get_controlled_joint_names()
    base_link = kinematics_parser.root_link

    output_dict = {}
    if "collision" in input_xrdf_dict:

        coll_name = return_value_if_exists(input_xrdf_dict["collision"], "geometry")

        if "spheres" not in input_xrdf_dict["geometry"][coll_name]:
            log_error("spheres key not found in xrdf")
        coll_spheres = return_value_if_exists(input_xrdf_dict["geometry"][coll_name], "spheres")
        output_dict["collision_spheres"] = coll_spheres

        buffer_distance = return_value_if_exists(
            input_xrdf_dict["collision"], "buffer_distance", raise_error=False
        )
        if buffer_distance is None:
            buffer_distance = 0.0
        output_dict["collision_sphere_buffer"] = buffer_distance
        output_dict["collision_link_names"] = list(coll_spheres.keys())

        if "self_collision" in input_xrdf_dict:
            if (
                input_xrdf_dict["self_collision"]["geometry"]
                != input_xrdf_dict["collision"]["geometry"]
            ):
                log_error("self_collision geometry does not match collision geometry")

            self_collision_ignore = return_value_if_exists(
                input_xrdf_dict["self_collision"],
                "ignore",
            )

            self_collision_buffer = return_value_if_exists(
                input_xrdf_dict["self_collision"],
                "buffer_distance",
                raise_error=False,
            )
            if self_collision_buffer is None:
                self_collision_buffer = {}
            output_dict["self_collision_ignore"] = self_collision_ignore
            output_dict["self_collision_buffer"] = self_collision_buffer
        else:
            log_error("self_collision key not found in xrdf")
    else:
        log_warn("collision key not found in xrdf, collision avoidance is disabled")

    tool_frames = return_value_if_exists(input_xrdf_dict, "tool_frames")

    output_dict["ee_link"] = tool_frames[0]
    if len(tool_frames) > 1:
        output_dict["link_names"] = deepcopy(tool_frames)

    # cspace:
    cspace_dict = return_value_if_exists(input_xrdf_dict, "cspace")

    active_joints = return_value_if_exists(cspace_dict, "joint_names")

    default_joint_positions = return_value_if_exists(input_xrdf_dict, "default_joint_positions")
    active_config = []
    locked_joints = {}

    for j in joint_names:
        if j in active_joints:
            if j in default_joint_positions:
                active_config.append(default_joint_positions[j])
            else:
                active_config.append(0.0)
        else:
            locked_joints[j] = 0.0
            if j in default_joint_positions:
                locked_joints[j] = default_joint_positions[j]

    acceleration_limits = return_value_if_exists(cspace_dict, "acceleration_limits")
    jerk_limits = return_value_if_exists(cspace_dict, "jerk_limits")

    max_acc = max(acceleration_limits)
    max_jerk = max(jerk_limits)
    output_dict["lock_joints"] = locked_joints
    all_joint_names = active_joints + list(locked_joints.keys())
    output_cspace = {
        "joint_names": all_joint_names,
        "retract_config": active_config + list(locked_joints.values()),
        "null_space_weight": [1.0 for _ in range(len(all_joint_names))],
        "cspace_distance_weight": [1.0 for _ in range(len(all_joint_names))],
        "max_acceleration": acceleration_limits
        + [max_acc for _ in range(len(all_joint_names) - len(active_joints))],
        "max_jerk": jerk_limits
        + [max_jerk for _ in range(len(all_joint_names) - len(active_joints))],
    }

    output_dict["cspace"] = output_cspace

    extra_links = {}
    if "modifiers" in input_xrdf_dict:
        for k in range(len(input_xrdf_dict["modifiers"])):
            mod_list = list(input_xrdf_dict["modifiers"][k].keys())
            if len(mod_list) > 1:
                log_error("Each modifier should have only one key")
                raise ValueError("Each modifier should have only one key")
            mod_type = mod_list[0]
            if mod_type == "set_base_frame":
                base_link = input_xrdf_dict["modifiers"][k]["set_base_frame"]
            elif mod_type == "add_frame":
                frame_data = input_xrdf_dict["modifiers"][k]["add_frame"]
                extra_links[frame_data["frame_name"]] = {
                    "parent_link_name": frame_data["parent_frame_name"],
                    "link_name": frame_data["frame_name"],
                    "joint_name": frame_data["joint_name"],
                    "joint_type": frame_data["joint_type"],
                    "fixed_transform": frame_data["fixed_transform"]["position"]
                    + [frame_data["fixed_transform"]["orientation"]["w"]]
                    + frame_data["fixed_transform"]["orientation"]["xyz"],
                }
            else:
                log_warn('XRDF modifier "' + mod_type + '" not recognized')
    output_dict["extra_links"] = extra_links

    output_dict["base_link"] = base_link

    output_dict["urdf_path"] = urdf_path

    output_dict = {"robot_cfg": {"kinematics": output_dict}}
    return output_dict
