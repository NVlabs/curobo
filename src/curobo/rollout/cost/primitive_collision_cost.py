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
from dataclasses import dataclass
from typing import Optional, Union

# Third Party
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionQueryBuffer, WorldCollision
from curobo.rollout.cost.cost_base import CostBase, CostConfig
from curobo.rollout.dynamics_model.integration_utils import interpolate_kernel, sum_matrix
from curobo.util.logger import log_info
from curobo.util.torch_utils import get_torch_jit_decorator


@dataclass
class PrimitiveCollisionCostConfig(CostConfig):
    """Create Collision Cost Configuration."""

    #: WorldCollision instance to use for distance queries.
    world_coll_checker: Optional[WorldCollision] = None

    #: Sweep for collisions between timesteps in a trajectory.
    use_sweep: bool = False
    use_sweep_kernel: bool = False
    sweep_steps: int = 4

    #: Speed metric scales the collision distance by sphere velocity (similar to CHOMP Planner
    #: ICRA'09). This prevents the optimizer from speeding through obstacles to minimize cost and
    #: instead encourages the robot to move around the obstacle.
    use_speed_metric: bool = False

    #: dt to use for computation of velocity and acceleration through central difference for
    #: speed metric. Value less than 1 is better as that leads to different scaling between
    #: acceleration and velocity.
    speed_dt: Union[torch.Tensor, float] = 0.01

    #: The distance outside collision at which to activate the cost. Having a non-zero value enables
    #: the robot to move slowly when within this distance to an obstacle. This enables our
    #: post optimization interpolation to not hit any obstacles.
    activation_distance: Union[torch.Tensor, float] = 0.0

    #: Setting this flag to true will sum the distance colliding obstacles.
    sum_collisions: bool = True

    #: Setting this flag to true will sum the distance across spheres of the robot.
    #: Setting to False will only take the max distance
    sum_distance: bool = True

    def __post_init__(self):
        if isinstance(self.speed_dt, float):
            self.speed_dt = self.tensor_args.to_device([self.speed_dt])
        if isinstance(self.activation_distance, float):
            self.activation_distance = self.tensor_args.to_device([self.activation_distance])
        return super().__post_init__()


class PrimitiveCollisionCost(CostBase, PrimitiveCollisionCostConfig):
    def __init__(self, config: PrimitiveCollisionCostConfig):
        """Creates a primitive collision cost instance.

        See note on :ref:`collision_checking_note` for details on the cost formulation.

        Args:
            config: Cost
        """

        PrimitiveCollisionCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        self._og_speed_dt = self.speed_dt.clone()
        self.batch_size = -1
        self._horizon = -1
        self._n_spheres = -1

        self.t_mat = None
        if self.classify:
            self.coll_check_fn = self.world_coll_checker.get_sphere_collision
            self.sweep_check_fn = self.world_coll_checker.get_swept_sphere_collision
        else:
            self.coll_check_fn = self.world_coll_checker.get_sphere_distance
            self.sweep_check_fn = self.world_coll_checker.get_swept_sphere_distance
        self.sampled_spheres = None
        self.sum_mat = None  #
        if self.use_sweep:
            # if self.use_sweep_kernel and (
            #    type(self.world_coll_checker) in [WorldMeshCollision, WorldPrimitiveCollision]
            # ):
            # TODO: Implement sweep for nvblox collision checker.
            self.forward = self.sweep_kernel_fn
            # else:
            # self.forward = self.discrete_fn
        else:
            self.forward = self.discrete_fn
        self.int_mat = None
        self._fd_matrix = None
        self._collision_query_buffer = CollisionQueryBuffer()

    def sweep_kernel_fn(self, robot_spheres_in, env_query_idx: Optional[torch.Tensor] = None):
        self._collision_query_buffer.update_buffer_shape(
            robot_spheres_in.shape, self.tensor_args, self.world_coll_checker.collision_types
        )
        if not self.sum_distance:
            log_info("sum_distance=False will be slower than sum_distance=True")
            self.return_loss = True
        dist = self.sweep_check_fn(
            robot_spheres_in,
            self._collision_query_buffer,
            self.weight,
            sweep_steps=self.sweep_steps,
            activation_distance=self.activation_distance,
            speed_dt=self.speed_dt,
            enable_speed_metric=self.use_speed_metric,
            env_query_idx=env_query_idx,
            return_loss=self.return_loss,
        )
        if self.classify:
            cost = weight_collision(dist, self.sum_distance)
        else:
            cost = weight_distance(dist, self.sum_distance)
        return cost

    def sweep_fn(self, robot_spheres_in, env_query_idx: Optional[torch.Tensor] = None):
        batch_size, horizon, n_spheres, _ = robot_spheres_in.shape
        # add intermediate spheres to account for discretization:
        new_horizon = (horizon - 1) * self.sweep_steps
        if self.int_mat is None:
            self.int_mat = interpolate_kernel(horizon, self.sweep_steps, self.tensor_args)
            self.int_mat_t = self.int_mat.transpose(0, 1)
            self.int_sum_mat = sum_matrix(horizon, self.sweep_steps, self.tensor_args)
        sampled_spheres = (
            (robot_spheres_in.transpose(1, 2).transpose(2, 3) @ self.int_mat_t)
            .transpose(2, 3)
            .transpose(1, 2)
            .contiguous()
        )
        # robot_spheres = sampled_spheres.view(batch_size * new_horizon * n_spheres, 4)
        # self.update_batch_size(batch_size * new_horizon * n_spheres)

        self._collision_query_buffer.update_buffer_shape(
            sampled_spheres.shape, self.tensor_args, self.world_coll_checker.collision_types
        )
        if not self.sum_distance:
            log_info("sum_distance=False will be slower than sum_distance=True")
            self.return_loss = True
        dist = self.coll_check_fn(
            sampled_spheres.contiguous(),
            self._collision_query_buffer,
            self.weight,
            activation_distance=self.activation_distance,
            env_query_idx=env_query_idx,
            return_loss=self.return_loss,
        )
        dist = dist.view(batch_size, new_horizon, n_spheres)

        if self.classify:
            cost = weight_sweep_collision(self.int_sum_mat, dist, self.sum_distance)
        else:
            cost = weight_sweep_distance(self.int_sum_mat, dist, self.sum_distance)

        return cost

    def discrete_fn(self, robot_spheres_in, env_query_idx: Optional[torch.Tensor] = None):
        self._collision_query_buffer.update_buffer_shape(
            robot_spheres_in.shape, self.tensor_args, self.world_coll_checker.collision_types
        )
        if not self.sum_distance:
            log_info("sum_distance=False will be slower than sum_distance=True")
            self.return_loss = True
        dist = self.coll_check_fn(
            robot_spheres_in,
            self._collision_query_buffer,
            self.weight,
            env_query_idx=env_query_idx,
            activation_distance=self.activation_distance,
            return_loss=self.return_loss,
            sum_collisions=self.sum_collisions,
        )

        if self.classify:
            cost = weight_collision(dist, self.sum_distance)
        else:
            cost = weight_distance(dist, self.sum_distance)
        return cost

    def update_dt(self, dt: Union[float, torch.Tensor]):
        self.speed_dt[:] = dt  # / self._og_speed_dt
        return super().update_dt(dt)

    def get_gradient_buffer(self):
        return self._collision_query_buffer.get_gradient_buffer()


@get_torch_jit_decorator()
def weight_sweep_distance(int_mat, dist, sum_cost: bool):
    if sum_cost:
        dist = torch.sum(dist, dim=-1)
    else:
        dist = torch.max(dist, dim=-1)[0]
    dist = dist @ int_mat
    return dist


@get_torch_jit_decorator()
def weight_sweep_collision(int_mat, dist, sum_cost: bool):
    if sum_cost:
        dist = torch.sum(dist, dim=-1)
    else:
        dist = torch.max(dist, dim=-1)[0]

    dist = torch.where(dist > 0, dist + 1.0, dist)
    dist = dist @ int_mat
    return dist


@get_torch_jit_decorator()
def weight_distance(dist, sum_cost: bool):
    if sum_cost:
        dist = torch.sum(dist, dim=-1)
    else:
        dist = torch.max(dist, dim=-1)[0]
    return dist


@get_torch_jit_decorator()
def weight_collision(dist, sum_cost: bool):
    if sum_cost:
        dist = torch.sum(dist, dim=-1)
    else:
        dist = torch.max(dist, dim=-1)[0]

    dist = torch.where(dist > 0, dist + 1.0, dist)
    return dist
