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
"""This example will use CuRobo's functions as pytorch layers as part of a NN."""
# Standard Library
import uuid
from typing import Optional

# Third Party
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# CuRobo
from curobo.geom.sdf.world import WorldConfig
from curobo.geom.types import Mesh
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_assets_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig


class CuroboTorch(torch.nn.Module):
    def __init__(self, robot_world: RobotWorld):
        """Build a simple structured NN:

        q_current -> kinematics -> sdf -> features
        [features, x_des] -> NN -> kinematics -> sdf -> [sdf, pose_distance] -> NN -> q_out
        loss = (fk(q_out) - x_des) + (q_current - q_out) + valid(q_out)
        """
        super(CuroboTorch, self).__init__()

        feature_dims = robot_world.kinematics.kinematics_config.link_spheres.shape[0] * 5 + 7 + 1
        q_feature_dims = 7
        final_feature_dims = feature_dims + 1 + 7
        output_dims = robot_world.kinematics.get_dof()

        # build neural network:
        self._robot_world = robot_world

        self._feature_mlp = nn.Sequential(
            nn.Linear(q_feature_dims, 512),
            nn.ReLU6(),
            nn.Linear(512, 512),
            nn.ReLU6(),
            nn.Linear(512, 512),
            nn.ReLU6(),
            nn.Linear(512, output_dims),
            nn.Tanh(),
        )
        self._final_mlp = nn.Sequential(
            nn.Linear(final_feature_dims, 256),
            nn.ReLU6(),
            nn.Linear(256, 256),
            nn.ReLU6(),
            nn.Linear(256, 64),
            nn.ReLU6(),
            nn.Linear(64, output_dims),
            nn.Tanh(),
        )

    def get_features(self, q: torch.Tensor, x_des: Optional[Pose] = None):
        kin_state = self._robot_world.get_kinematics(q)
        spheres = kin_state.link_spheres_tensor.unsqueeze(2)
        q_sdf = self._robot_world.get_collision_distance(spheres)
        q_self = self._robot_world.get_self_collision_distance(
            kin_state.link_spheres_tensor.unsqueeze(1)
        )

        features = [
            kin_state.link_spheres_tensor.view(q.shape[0], -1),
            q_sdf,
            q_self,
            kin_state.ee_position,
            kin_state.ee_quaternion,
        ]
        if x_des is not None:
            pose_distance = self._robot_world.pose_distance(
                x_des, kin_state.ee_pose, resize=True
            ).view(-1, 1)
            features.append(pose_distance)
            features.append(x_des.position)
            features.append(x_des.quaternion)

        features = torch.cat(features, dim=-1)

        return features

    def forward(self, q: torch.Tensor, x_des: Pose):
        """Forward for neural network

        Args:
            q (torch.Tensor): _description_
            x_des (torch.Tensor): _description_
        """
        # get features for input:
        in_features = torch.cat([x_des.position, x_des.quaternion], dim=-1)
        # pass through initial mlp:
        q_mid = self._feature_mlp(in_features)

        q_scale = self._robot_world.bound_scale * q_mid
        # get new features:
        mid_features = self.get_features(q_scale, x_des=x_des)
        q_out = self._final_mlp(mid_features)
        q_out = self._robot_world.bound_scale * q_out
        return q_out

    def loss(self, x_des: Pose, q: torch.Tensor, q_in: torch.Tensor):
        kin_state = self._robot_world.get_kinematics(q)
        distance = self._robot_world.pose_distance(x_des, kin_state.ee_pose, resize=True)
        d_sdf = self._robot_world.collision_constraint(
            kin_state.link_spheres_tensor.unsqueeze(1)
        ).view(-1)
        d_self = self._robot_world.self_collision_cost(
            kin_state.link_spheres_tensor.unsqueeze(1)
        ).view(-1)
        loss = 0.1 * torch.linalg.norm(q_in - q, dim=-1) + distance + 100.0 * (d_self + d_sdf)
        return loss

    def val_loss(self, x_des: Pose, q: torch.Tensor, q_in: torch.Tensor):
        kin_state = self._robot_world.get_kinematics(q)
        distance = self._robot_world.pose_distance(x_des, kin_state.ee_pose, resize=True)
        d_sdf = self._robot_world.collision_constraint(
            kin_state.link_spheres_tensor.unsqueeze(1)
        ).view(-1)
        d_self = self._robot_world.self_collision_cost(
            kin_state.link_spheres_tensor.unsqueeze(1)
        ).view(-1)
        loss = 10.0 * (d_self + d_sdf) + distance
        return loss


if __name__ == "__main__":
    update_goal = False
    write_usd = False
    writer = SummaryWriter("log/runs/ik/" + str(uuid.uuid4()))
    robot_file = "franka.yml"
    world_file = "collision_table.yml"
    config = RobotWorldConfig.load_from_config(robot_file, world_file, pose_weight=[10, 200, 1, 10])
    curobo_fn = RobotWorld(config)

    model = CuroboTorch(curobo_fn)
    model.cuda()
    with torch.no_grad():
        # q_train = curobo_fn.sample(10000)
        q_val = curobo_fn.sample(100)
        q_train = curobo_fn.sample(5 * 2048)
    usd_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    batch_size = 512
    batch_start = torch.arange(0, q_train.shape[0], batch_size)
    with torch.no_grad():
        x_des = curobo_fn.get_kinematics(q_train[1:2]).ee_pose
        x_des_train = curobo_fn.get_kinematics(q_train).ee_pose
        x_des_val = x_des.repeat(q_val.shape[0])
    q_debug = []
    bar = tqdm(range(500))
    for e in bar:
        model.train()
        for j in range(batch_start.shape[0]):
            x_train = q_train[batch_start[j] : batch_start[j] + batch_size]
            if x_train.shape[0] != batch_size:
                continue
            x_des_batch = x_des_train[batch_start[j] : batch_start[j] + batch_size]
            q = model.forward(x_train, x_des_batch)
            loss = model.loss(x_des_batch, q, x_train)
            loss = torch.mean(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        writer.add_scalar("training loss", loss.item(), e)
        model.eval()
        with torch.no_grad():
            q_pred = model.forward(q_val, x_des_val)
            val_loss = model.val_loss(x_des_val, q_pred, q_val)
            val_loss = torch.mean(val_loss)
            q_debug.append(q_pred[0:1].clone())
        writer.add_scalar("validation loss", val_loss.item(), e)
        bar.set_description("t: " + str(val_loss.item()))
        if e % 100 == 0 and len(q_debug) > 1:
            if write_usd:
                q_traj = torch.cat(q_debug, dim=0)
                world_model = WorldConfig.from_dict(
                    load_yaml(join_path(get_world_configs_path(), world_file))
                )
                gripper_mesh = Mesh(
                    name="target_gripper",
                    file_path=join_path(
                        get_assets_path(),
                        "robot/franka_description/meshes/visual/hand_ee_link.dae",
                    ),
                    color=[0.0, 0.8, 0.1, 1.0],
                    pose=x_des[0].tolist(),
                )
                world_model.add_obstacle(gripper_mesh)
                save_name = "e_" + str(e)
                UsdHelper.write_trajectory_animation_with_robot_usd(
                    "franka.yml",
                    world_model,
                    JointState(position=q_traj[0]),
                    JointState(position=q_traj),
                    dt=1.0,
                    visualize_robot_spheres=False,
                    save_path=save_name + ".usd",
                    base_frame="/" + save_name,
                )
                usd_list.append(save_name + ".usd")
            if update_goal:
                with torch.no_grad():
                    rand_perm = torch.randperm(q_val.shape[0])
                    q_val = q_val[rand_perm].clone()
                    x_des = curobo_fn.get_kinematics(q_val[0:1]).ee_pose
                    x_des_val = x_des.repeat(q_val.shape[0])
            q_debug = []

    # create loss function:
    if write_usd:
        UsdHelper.create_grid_usd(
            usd_list,
            "epoch_grid.usd",
            "/world",
            max_envs=len(usd_list),
            max_timecode=len(q_debug),
            x_space=2.0,
            y_space=2.0,
            x_per_row=int(np.sqrt(len(usd_list))) + 1,
            local_asset_path="",
            dt=1,
        )
