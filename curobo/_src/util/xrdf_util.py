# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Standard Library
from copy import deepcopy
from typing import Any, Dict, Optional

# CuRobo
from curobo._src.robot.parser.parser_urdf import UrdfRobotParser
from curobo._src.types.content_path import ContentPath
from curobo._src.util.logging import log_and_raise, log_warn
from curobo._src.util_file import load_yaml


def return_value_if_exists(
    input_dict: Dict, key: str, suffix: str = "xrdf", raise_error: bool = True
) -> Any:
    if key not in input_dict:
        if raise_error:
            log_and_raise(key + " key not found in " + suffix)
        return None
    return input_dict[key]


def convert_xrdf_to_curobo(
    content_path: ContentPath = ContentPath(),
    input_xrdf_dict: Optional[Dict] = None,
) -> Dict:
    if content_path.robot_urdf_absolute_path is None:
        log_and_raise(
            "content_path.robot_urdf_absolute_path or content_path.robot_urdf_file \
                 is required."
        )
    urdf_path = content_path.robot_urdf_absolute_path
    if input_xrdf_dict is None:
        input_xrdf_dict = load_yaml(content_path.robot_xrdf_absolute_path)
    if input_xrdf_dict is None:
        log_and_raise("Failed to load XRDF file from " + str(content_path.robot_xrdf_absolute_path))

    if isinstance(content_path, str):
        log_and_raise("content_path should be of type ContentPath")

    if return_value_if_exists(input_xrdf_dict, "format") != "xrdf":
        log_and_raise("format is not xrdf")

    if return_value_if_exists(input_xrdf_dict, "format_version") > 1.0:
        log_warn("format_version is greater than 1.0")
    # Also get base link as root of urdf
    kinematics_parser = UrdfRobotParser(
        urdf_path, mesh_root=content_path.robot_asset_absolute_path, build_scene_graph=True
    )
    joint_names = kinematics_parser.get_controlled_joint_names()
    base_link = kinematics_parser.root_link

    output_dict = {}
    if "collision" in input_xrdf_dict:
        coll_name = return_value_if_exists(input_xrdf_dict["collision"], "geometry")

        if "spheres" not in input_xrdf_dict["geometry"][coll_name]:
            log_and_raise("spheres key not found in xrdf")
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
                log_and_raise("self_collision geometry does not match collision geometry")

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
            log_and_raise("self_collision key not found in xrdf")
    else:
        log_warn("collision key not found in xrdf, collision avoidance is disabled")

    tool_frames = return_value_if_exists(input_xrdf_dict, "tool_frames")


    output_dict["tool_frames"] = deepcopy(tool_frames)

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
        "default_joint_position": active_config + list(locked_joints.values()),
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
                log_and_raise("Each modifier should have only one key")
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

    if "dynamics" in input_xrdf_dict:
        output_dict["robot_cfg"]["dynamics"] = input_xrdf_dict["dynamics"]
    return output_dict




def convert_curobo_to_xrdf(
    input_curobo_dict: Dict,
    geometry_name: str = "collision_model",
) -> Dict:
    """Convert a CuRobo-style kinematics configuration dictionary back to an XRDF dictionary.

    The function accepts either the full CuRobo config with the envelope
    {"robot_cfg": {"kinematics": {...}}} or just the inner kinematics dict.

    Parameters
    ----------
    input_curobo_dict: Dict
        CuRobo configuration dict or the inner kinematics dict.
    geometry_name: str
        Name to use for the geometry entry that will hold collision spheres.

    Returns:
    -------
    Dict
        XRDF dictionary with keys like: format, format_version, geometry,
        collision, self_collision, tool_frames, cspace, default_joint_positions,
        and modifiers.
    """
    # Unwrap to kinematics dict
    # Unwrap extra "robot_cfg" field if needed
    if "robot_cfg" in input_curobo_dict:
        kinematics_dict = return_value_if_exists(
            input_curobo_dict["robot_cfg"], "kinematics"
        )
    else:
        kinematics_dict = return_value_if_exists(input_curobo_dict, "kinematics")

    if not isinstance(kinematics_dict, dict) or not kinematics_dict:
        log_and_raise("\"robot_cfg/kinematics\" nor \"kinematics\" keys not found in input_curobo_dict")
        return {}

    xrdf: Dict[str, Any] = {
        "format": "xrdf",
        "format_version": 1.0,
    }

    # Tool frames
    tool_frames_val = return_value_if_exists(kinematics_dict, "tool_frames")
    if tool_frames_val is not None:
        xrdf["tool_frames"] = tool_frames_val

    # Collision geometry
    geometry: Dict[str, Any] = {}
    collision_section: Dict[str, Any] = {"geometry": geometry_name}

    if "collision_spheres" in kinematics_dict:
        coll_spheres = return_value_if_exists(kinematics_dict, "collision_spheres")
        if coll_spheres is not None:
            if isinstance(coll_spheres, str):
                coll_spheres = load_yaml(coll_spheres)
        geometry[geometry_name] = {"spheres": deepcopy(coll_spheres) if coll_spheres else {}}
        # Optional buffer on collision
        buffer_val = return_value_if_exists(kinematics_dict, "collision_sphere_buffer")
        collision_section["buffer_distance"] = buffer_val if buffer_val is not None else 0.0
    else:
        log_warn("No collision_spheres in CuRobo config; collision section will be minimal")
        geometry[geometry_name] = {"spheres": {}}
        collision_section["buffer_distance"] = 0.0

    xrdf["geometry"] = geometry
    xrdf["collision"] = collision_section

    # Self-collision
    self_collision_section: Dict[str, Any] = {"geometry": geometry_name}
    sc_ignore = return_value_if_exists(kinematics_dict, "self_collision_ignore")
    self_collision_section["ignore"] = deepcopy(sc_ignore) if sc_ignore is not None else []

    sc_buffer = return_value_if_exists(kinematics_dict, "self_collision_buffer")
    self_collision_section["buffer_distance"] = deepcopy(sc_buffer) if sc_buffer is not None else {}

    xrdf["self_collision"] = self_collision_section

    # C-space reconstruction
    cspace_in = return_value_if_exists(kinematics_dict, "cspace")
    all_joint_names = return_value_if_exists(cspace_in, "joint_names")
    default_joint_position = return_value_if_exists(cspace_in, "default_joint_position")
    max_acc_all = return_value_if_exists(cspace_in, "max_acceleration")
    max_jerk_all = return_value_if_exists(cspace_in, "max_jerk")
    lock_joints = return_value_if_exists(kinematics_dict, "lock_joints")

    if not isinstance(all_joint_names, list) or not isinstance(default_joint_position, list):
        log_and_raise("cspace.joint_names or cspace.default_joint_position missing/invalid in CuRobo config")
        return xrdf

    if len(all_joint_names) != len(default_joint_position):
        log_warn("Length mismatch between cspace.joint_names and cspace.default_joint_position")

    # Active joints are those not present in lock_joints
    if lock_joints is not None:
        active_joints = [j for j in all_joint_names if j not in lock_joints]
    else:
        active_joints = all_joint_names

    # Build default joint positions from default_joint_position for all joints
    default_joint_positions: Dict[str, float] = {}
    for idx, name in enumerate(all_joint_names):
        if idx < len(default_joint_position):
            default_joint_positions[name] = default_joint_position[idx]
        else:
            # Fallback if default_joint_position is short; prefer lock_joints where available
            default_joint_positions[name] = lock_joints.get(name, 0.0)

    # Acceleration and jerk for active joints only (XRDF expects per-active-joint)
    name_to_index = {n: i for i, n in enumerate(all_joint_names)}
    acceleration_limits = []
    jerk_limits = []
    if isinstance(max_acc_all, float):
        max_acc_all = [max_acc_all] * len(active_joints)
    if isinstance(max_jerk_all, float):
        max_jerk_all = [max_jerk_all] * len(active_joints)
    for n in active_joints:
        idx = name_to_index.get(n, -1)
        if 0 <= idx < len(max_acc_all):
            acceleration_limits.append(max_acc_all[idx])
        else:
            # Fallback: use max over provided values or 0.0
            acceleration_limits.append(max(max_acc_all) if max_acc_all else 0.0)
        if 0 <= idx < len(max_jerk_all):
            jerk_limits.append(max_jerk_all[idx])
        else:
            jerk_limits.append(max(max_jerk_all) if max_jerk_all else 0.0)

    xrdf["cspace"] = {
        "joint_names": active_joints,
        "acceleration_limits": acceleration_limits,
        "jerk_limits": jerk_limits,
    }

    # Modifiers (base_link and extra_links)
    modifiers = []
    base_link = return_value_if_exists(kinematics_dict, "base_link", raise_error=False)
    if base_link is not None:
        modifiers.append({"set_base_frame": base_link})

    extra_links = return_value_if_exists(kinematics_dict, "extra_links", raise_error=False)
    if extra_links is not None:
        for frame_name, frame_data in extra_links.items():
            # Extract position and orientation from fixed_transform
            # fixed_transform format: [x, y, z, w, qx, qy, qz]
            ft = frame_data.get("fixed_transform", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            modifiers.append({
                "add_frame": {
                    "frame_name": frame_data.get("link_name", frame_name),
                    "parent_frame_name": frame_data.get("parent_link_name", ""),
                    "joint_name": frame_data.get("joint_name", ""),
                    "joint_type": frame_data.get("joint_type", "fixed"),
                    "fixed_transform": {
                        "position": ft[:3],
                        "orientation": {
                            "w": ft[3],
                            "xyz": ft[4:7],
                        },
                    },
                }
            })

    if modifiers:
        xrdf["modifiers"] = modifiers

    xrdf["default_joint_positions"] = default_joint_positions

    return xrdf
