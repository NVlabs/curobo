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
from typing import Dict, List

# Third Party
import numpy as np

# CuRobo
from curobo.util.logger import log_warn
from curobo.util.usd_helper import set_prim_transform
from matplotlib import cm
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid
from omni.isaac.core.robots import Robot

# USD imports for plant physics manipulation
from pxr import PhysxSchema, UsdPhysics

ISAAC_SIM_23 = False
ISAAC_SIM_45 = False
try:
    # Third Party
    from omni.isaac.urdf import _urdf  # isaacsim 2022.2
except ImportError:
    # Third Party
    try:
        from omni.importer.urdf import _urdf  # isaac sim 2023.1 or above
    except ImportError:
        from isaacsim.asset.importer.urdf import _urdf  # isaac sim 4.5+

        ISAAC_SIM_45 = True
    ISAAC_SIM_23 = True

try:
    # import for older isaacsim installations
    from omni.isaac.core.materials import OmniPBR
except ImportError:
    # import for isaac sim 4.5+
    from isaacsim.core.api.materials import OmniPBR


# Standard Library
from typing import Optional

# CuRobo
from curobo.util_file import get_assets_path, get_filename, get_path_of_dir, join_path

# Third Party
from omni.isaac.core.utils.extensions import enable_extension


def add_extensions(simulation_app, headless_mode: Optional[str] = None):
    ext_list = [
        "omni.kit.asset_converter",
        "omni.kit.tool.asset_importer",
        "omni.isaac.asset_browser",
    ]
    if headless_mode is not None:
        log_warn("Running in headless mode: " + headless_mode)
        ext_list += ["omni.kit.livestream." + headless_mode]
    [enable_extension(x) for x in ext_list]
    simulation_app.update()

    return True


############################################################
def add_robot_to_scene(
    robot_config: Dict,
    my_world: World,
    load_from_usd: bool = False,
    subroot: str = "",
    robot_name: str = "robot",
    position: np.array = np.array([0, 0, 0]),
    initialize_world: bool = True,
):
    urdf_interface = _urdf.acquire_urdf_interface()
    # Set the settings in the import config
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = False
    import_config.default_drive_strength = 1047.19751
    import_config.default_position_drive_damping = 52.35988
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0

    asset_path = get_assets_path()
    if (
        "external_asset_path" in robot_config["kinematics"]
        and robot_config["kinematics"]["external_asset_path"] is not None
    ):
        asset_path = robot_config["kinematics"]["external_asset_path"]

    # urdf_path:
    # meshes_path:
    # meshes path should be a subset of urdf_path
    full_path = join_path(asset_path, robot_config["kinematics"]["urdf_path"])
    # full path contains the path to urdf
    # Get meshes path
    robot_path = get_path_of_dir(full_path)
    filename = get_filename(full_path)
    if ISAAC_SIM_45:
        import omni.kit.commands
        import omni.usd

        from isaacsim.core.utils.extensions import get_extension_path_from_name

        # Retrieve the path of the URDF file from the extension
        extension_path = get_extension_path_from_name("isaacsim.asset.importer.urdf")
        root_path = robot_path
        file_name = filename

        # Parse the robot's URDF file to generate a robot model

        dest_path = join_path(
            root_path, get_filename(file_name, remove_extension=True) + "_temp.usd"
        )

        result, robot_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path="{}/{}".format(root_path, file_name),
            import_config=import_config,
            dest_path=dest_path,
        )
        prim_path = omni.usd.get_stage_next_free_path(
            my_world.scene.stage,
            str(my_world.scene.stage.GetDefaultPrim().GetPath()) + robot_path,
            False,
        )
        robot_prim = my_world.scene.stage.OverridePrim(prim_path)
        robot_prim.GetReferences().AddReference(dest_path)
        robot_path = prim_path
    else:
        imported_robot = urdf_interface.parse_urdf(robot_path, filename, import_config)
        dest_path = subroot

        robot_path = urdf_interface.import_robot(
            robot_path,
            filename,
            imported_robot,
            import_config,
            dest_path,
        )

    base_link_name = robot_config["kinematics"]["base_link"]

    robot_p = Robot(
        prim_path=robot_path + "/" + base_link_name,
        name=robot_name,
    )

    robot_prim = robot_p.prim
    stage = robot_prim.GetStage()
    linkp = stage.GetPrimAtPath(robot_path)
    set_prim_transform(linkp, [position[0], position[1], position[2], 1, 0, 0, 0])

    robot = my_world.scene.add(robot_p)
    if initialize_world:
        if ISAAC_SIM_45:
            my_world.initialize_physics()
            robot.initialize()

    return robot, robot_path


def configure_prim_physics(prim) -> None:
    """Configure physics properties for a single prim and its children."""
    if prim.IsValid():
        # Disable collisions
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        collision_api.GetCollisionEnabledAttr().Set(False)
        print(f"DISABLING collision for {prim.GetPath().pathString}")

        # Disable rigid body physics if present
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
            rigid_body_api.GetRigidBodyEnabledAttr().Set(False)
            print(f"DISABLING rigid body for {prim.GetPath().pathString}")

        # Also disable PhysX-specific rigid body properties if present
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            # Disable gravity as well to ensure no physics forces
            physx_rigid_body_api.GetDisableGravityAttr().Set(True)
            print(
                f"DISABLING PhysX rigid body properties for {prim.GetPath().pathString}"
            )

    # Recursively apply to all children
    for child in prim.GetAllChildren():
        configure_prim_physics(child)


def apply_plant_physics_settings(plant_prim) -> None:
    """Apply physics settings for plant stability by disabling physics for all plant components."""
    if not plant_prim or not plant_prim.IsValid():
        print("WARNING: Invalid plant prim provided to apply_plant_physics_settings")
        return

    print(f"Applying physics settings to plant: {plant_prim.GetPath().pathString}")

    # Apply physics settings to all children of the plant
    for prim in plant_prim.GetAllChildren():
        configure_prim_physics(prim)


def disable_physics_for_all_plants(stage, plant_scene_paths: List[str] = None) -> None:
    """Walk through all plants in the scene and disable their physics.

    Args:
        stage: The USD stage
        plant_scene_paths: List of plant scene paths. If None, defaults to common plant paths.
    """
    if plant_scene_paths is None:
        plant_scene_paths = [
            "/World/PlantScene",
            "/World/plant_004",
            "/World/plants",
        ]

    plants_found = 0

    for plant_path in plant_scene_paths:
        plant_prim = stage.GetPrimAtPath(plant_path)
        if plant_prim and plant_prim.IsValid():
            print(f"Found plant scene at: {plant_path}")
            apply_plant_physics_settings(plant_prim)
            plants_found += 1
        else:
            # Check if this path exists as a substring in any stage prims
            for prim in stage.Traverse():
                if (
                    plant_path.split("/")[-1].lower()
                    in prim.GetPath().pathString.lower()
                ):
                    if "plant" in prim.GetPath().pathString.lower():
                        print(f"Found plant-like prim at: {prim.GetPath().pathString}")
                        apply_plant_physics_settings(prim)
                        plants_found += 1
                        break

    if plants_found > 0:
        print(f"✅ Successfully disabled physics for {plants_found} plant scene(s)")
    else:
        print("⚠️ No plant scenes found to disable physics for")


class VoxelManager:
    def __init__(
        self,
        num_voxels: int = 5000,
        size: float = 0.02,
        color: List[float] = [1, 1, 1],
        prefix_path: str = "/World/curobo/voxel_",
        material_path: str = "/World/looks/v_",
    ) -> None:
        self.cuboid_list = []
        self.cuboid_material_list = []
        self.disable_idx = num_voxels
        for i in range(num_voxels):
            target_material = OmniPBR("/World/looks/v_" + str(i), color=np.ravel(color))

            cube = cuboid.VisualCuboid(
                prefix_path + str(i),
                position=np.array([0, 0, -10]),
                orientation=np.array([1, 0, 0, 0]),
                size=size,
                visual_material=target_material,
            )
            self.cuboid_list.append(cube)
            self.cuboid_material_list.append(target_material)
            cube.set_visibility(True)

    def update_voxels(self, voxel_position: np.ndarray, color_axis: int = 0):
        max_index = min(voxel_position.shape[0], len(self.cuboid_list))

        jet = cm.get_cmap("hot")  # .reversed()
        z_val = voxel_position[:, 0]

        jet_colors = jet(z_val)

        for i in range(max_index):
            self.cuboid_list[i].set_visibility(True)

            self.cuboid_list[i].set_local_pose(translation=voxel_position[i])
            self.cuboid_material_list[i].set_color(jet_colors[i][:3])

        for i in range(max_index, len(self.cuboid_list)):
            self.cuboid_list[i].set_local_pose(translation=np.ravel([0, 0, -10.0]))

            # self.cuboid_list[i].set_visibility(False)

    def clear(self):
        for i in range(len(self.cuboid_list)):
            self.cuboid_list[i].set_local_pose(translation=np.ravel([0, 0, -10.0]))


def visualize_approach_waypoints(
    start_pos, target_pos, approach_vector, step_size, num_steps=5
):
    """Visualize the expected approach waypoints for debugging."""
    from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_lines()

    # Draw the approach vector
    approach_line_length = 0.2  # 20cm
    draw.draw_lines(
        [start_pos],
        [start_pos + approach_vector * approach_line_length],
        [(0, 0, 1, 1)],  # Blue for approach vector
        [5.0],
    )

    # Draw line from start to target
    draw.draw_lines(
        [start_pos],
        [target_pos],
        [(0, 1, 0, 0.5)],  # Green for direct path
        [2.0],
    )

    # Draw intermediate waypoints
    waypoints = []
    positions_start = []
    positions_end = []
    colors = []
    sizes = []

    # Generate waypoints along approach vector
    for i in range(1, num_steps + 1):
        waypoint = start_pos + approach_vector * (step_size * i)
        waypoints.append(waypoint)

        # Add to visualization lists
        if i > 1:
            positions_start.append(waypoints[i - 2])
            positions_end.append(waypoint)
            colors.append((1, 0, 1, 0.8))  # Purple for waypoint connections
            sizes.append(3.0)

    # Draw waypoint lines
    if positions_start:
        draw.draw_lines(positions_start, positions_end, colors, sizes)

    # Draw waypoint spheres
    for i, waypoint in enumerate(waypoints):
        # Different color for last waypoint
        color = (1, 0, 0, 1) if i == len(waypoints) - 1 else (1, 1, 0, 1)
        draw.draw_points([waypoint], [color], [10.0])

    # Return the waypoints for further inspection
    return waypoints


def visualize_tool_axes(position, rotation_matrix, scale=0.1):
    """Visualize the tool coordinate axes at a given position and orientation."""
    from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_lines()

    # Extract the axes from rotation matrix
    x_axis = rotation_matrix[:, 0] * scale
    y_axis = rotation_matrix[:, 1] * scale
    z_axis = rotation_matrix[:, 2] * scale

    # Draw the three axes
    # X-axis (red)
    draw.draw_lines(
        [position],
        [position + x_axis],
        [(1, 0, 0, 1)],  # Red
        [5.0],
    )

    # Y-axis (green)
    draw.draw_lines(
        [position],
        [position + y_axis],
        [(0, 1, 0, 1)],  # Green
        [5.0],
    )

    # Z-axis (blue)
    draw.draw_lines(
        [position],
        [position + z_axis],
        [(0, 0, 1, 1)],  # Blue
        [5.0],
    )
