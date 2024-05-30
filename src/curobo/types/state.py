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
from __future__ import annotations

# Standard Library
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

# Third Party
import numpy as np
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.tensor import T_BDOF, T_DOF
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util.tensor_util import (
    check_tensor_shapes,
    clone_if_not_none,
    copy_tensor,
    fd_tensor,
    tensor_repeat_seeds,
)
from curobo.util.torch_utils import get_torch_jit_decorator


@dataclass
class FilterCoeff:
    position: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    jerk: float = 0.0


@dataclass
class State(Sequence):
    def blend(self, coeff: FilterCoeff, new_state: State):
        return self

    def to(self, tensor_args: TensorDeviceType):
        return self

    def get_state_tensor(self):
        return torch.tensor([0.0])

    def apply_kernel(self, kernel_mat):
        return self

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError


@dataclass
class JointState(State):
    position: Union[List[float], T_DOF]
    velocity: Union[List[float], T_DOF, None] = None
    acceleration: Union[List[float], T_DOF, None] = None
    joint_names: Optional[List[str]] = None
    jerk: Union[List[float], T_DOF, None] = None  # Optional
    tensor_args: TensorDeviceType = TensorDeviceType()
    aux_data: dict = field(default_factory=lambda: {})

    def __post_init__(self):
        if isinstance(self.position, torch.Tensor):
            self.tensor_args = TensorDeviceType(self.position.device)

    @staticmethod
    def from_numpy(
        joint_names: List[str],
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        acceleration: Optional[np.ndarray] = None,
        jerk: Optional[np.ndarray] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        pos = tensor_args.to_device(position)
        vel = acc = je = None
        if velocity is not None:
            vel = tensor_args.to_device(velocity)
        else:
            vel = pos * 0.0
        if acceleration is not None:
            acc = tensor_args.to_device(acceleration)
        else:
            acc = pos * 0.0
        if jerk is not None:
            je = tensor_args.to_device(jerk)
        else:
            je = pos * 0.0
        return JointState(pos, vel, acc, joint_names=joint_names, jerk=je, tensor_args=tensor_args)

    @staticmethod
    def from_position(position: T_BDOF, joint_names: Optional[List[str]] = None):
        return JointState(
            position=position,
            velocity=position * 0.0,
            acceleration=position * 0.0,
            jerk=position * 0.0,
            joint_names=joint_names,
        )

    def apply_kernel(self, kernel_mat):
        return JointState(
            position=kernel_mat @ self.position,
            velocity=kernel_mat @ self.velocity,
            acceleration=kernel_mat @ self.acceleration,
            joint_names=self.joint_names,
        )

    def repeat_seeds(self, num_seeds: int):
        return JointState(
            position=tensor_repeat_seeds(self.position, num_seeds),
            velocity=(
                tensor_repeat_seeds(self.velocity, num_seeds) if self.velocity is not None else None
            ),
            acceleration=(
                tensor_repeat_seeds(self.acceleration, num_seeds)
                if self.acceleration is not None
                else None
            ),
            joint_names=self.joint_names,
        )

    def to(self, tensor_args: TensorDeviceType):
        position = tensor_args.to_device(self.position)
        velocity = acceleration = jerk = None
        if self.velocity is not None:
            velocity = tensor_args.to_device(self.velocity)
        if self.acceleration is not None:
            acceleration = tensor_args.to_device(self.acceleration)
        if self.jerk is not None:
            jerk = tensor_args.to_device(self.jerk)
        return JointState(
            position,
            velocity,
            acceleration,
            jerk=jerk,
            tensor_args=tensor_args,
            joint_names=self.joint_names,
        )

    def clone(self):
        j_names = None
        if self.joint_names is not None:
            j_names = self.joint_names.copy()
        return JointState(
            position=clone_if_not_none(self.position),
            velocity=clone_if_not_none(self.velocity),
            acceleration=clone_if_not_none(self.acceleration),
            jerk=clone_if_not_none(self.jerk),
            joint_names=j_names,
            tensor_args=self.tensor_args,
        )

    def blend(self, coeff: FilterCoeff, new_state: JointState):
        self.position[:] = (
            coeff.position * new_state.position + (1.0 - coeff.position) * self.position
        )
        self.velocity[:] = (
            coeff.velocity * new_state.velocity + (1.0 - coeff.velocity) * self.velocity
        )
        self.acceleration[:] = (
            coeff.acceleration * new_state.acceleration
            + (1.0 - coeff.acceleration) * self.acceleration
        )
        self.jerk[:] = coeff.jerk * new_state.jerk + (1.0 - coeff.jerk) * self.jerk
        return self

    def get_state_tensor(self):
        velocity = self.velocity
        acceleration = self.acceleration
        jerk = self.jerk
        if velocity is None:
            velocity = self.position * 0.0
        if acceleration is None:
            acceleration = self.position * 0.0
        if jerk is None:
            jerk = self.position * 0.0
        state_tensor = torch.cat((self.position, velocity, acceleration, jerk), dim=-1)
        return state_tensor

    @staticmethod
    def from_state_tensor(state_tensor, joint_names=None, dof=7):
        return JointState(
            state_tensor[..., :dof],
            state_tensor[..., dof : 2 * dof],
            state_tensor[..., 2 * dof : 3 * dof],
            jerk=state_tensor[..., 3 * dof : 4 * dof],
            joint_names=joint_names,
        )

    def stack(self, new_state: JointState):
        return JointState.from_state_tensor(
            torch.cat((self.get_state_tensor(), new_state.get_state_tensor()), dim=-2),
            joint_names=self.joint_names,
            dof=self.position.shape[-1],
        )

    def __getitem__(self, idx):
        j = None
        v = a = None
        max_idx = 0
        if isinstance(idx, List):
            idx = torch.as_tensor(idx, device=self.position.device, dtype=torch.long)
        if isinstance(idx, int):
            max_idx = idx
        elif isinstance(idx, torch.Tensor):
            max_idx = torch.max(idx)
        if max_idx >= self.position.shape[0]:
            raise ValueError(
                str(max_idx)
                + " index out of range, current state is of length "
                + str(self.position.shape)
            )
        if isinstance(idx, int):
            p, v, a, j = jit_get_index_int(
                self.position, self.velocity, self.acceleration, self.jerk, idx
            )
        elif isinstance(idx, torch.Tensor):
            p, v, a, j = jit_get_index(
                self.position, self.velocity, self.acceleration, self.jerk, idx
            )
        else:
            p, v, a, j = fn_get_index(
                self.position, self.velocity, self.acceleration, self.jerk, idx
            )

        return JointState(p, v, a, joint_names=self.joint_names, jerk=j)

    def __len__(self):
        return self.position.shape[0]

    @staticmethod
    def from_list(position, velocity, acceleration, tensor_args: TensorDeviceType()):
        js = JointState(position, velocity, acceleration)
        js = js.to(tensor_args)
        return js

    def copy_at_index(self, in_joint_state: JointState, idx: Union[int, torch.Tensor]):
        """Copy joint state to specific index

        Args:
            in_joint_state (JointState): _description_
            idx (Union[int,torch.Tensor]): _description_
        """
        max_idx = 0
        if isinstance(idx, int):
            max_idx = idx
        elif isinstance(idx, List):
            max_idx = max(idx)
        elif isinstance(idx, torch.Tensor):
            max_idx = torch.max(idx)
        if self.position is not None:
            if max_idx >= self.position.shape[0]:
                raise ValueError(
                    str(max_idx)
                    + " index out of range, current state is of length "
                    + str(self.position.shape[0])
                )
            self.position[idx] = in_joint_state.position
        if self.velocity is not None:
            self.velocity[idx] = in_joint_state.velocity
        if self.acceleration is not None:
            self.acceleration[idx] = in_joint_state.acceleration
        if self.jerk is not None:
            self.jerk[idx] = in_joint_state.jerk

    def copy_data(self, in_joint_state: JointState):
        """Copy data from in_joint_state to self

        Args:
            in_joint_state (JointState): _description_
        """
        log_warn("JointState.copy_data is deprecated, use JointState.copy_ instead")
        if not copy_tensor(in_joint_state.position, self.position):
            self.position = in_joint_state.position
            log_info("Cloning JointState")
            print(self.position.shape, in_joint_state.position.shape)

        if not copy_tensor(in_joint_state.velocity, self.velocity):
            self.velocity = in_joint_state.velocity
        if not copy_tensor(in_joint_state.acceleration, self.acceleration):
            self.acceleration = in_joint_state.acceleration
        return self

    def _same_shape(self, new_js: JointState):
        same_shape = False

        if (
            check_tensor_shapes(new_js.position, self.position)
            and check_tensor_shapes(new_js.velocity, self.velocity)
            and check_tensor_shapes(new_js.acceleration, self.acceleration)
        ):
            same_shape = True

        # optional jerk check:
        # if self.jerk is not None and new_js.jerk is not None:
        #    same_shape = same_shape and check_tensor_shapes(new_js.jerk, self.jerk)
        return same_shape

    def copy_(self, in_joint_state: JointState):
        # return self.copy_data(in_joint_state)
        # copy data if tensor shapes are same:
        if in_joint_state.joint_names is not None:
            self.joint_names = in_joint_state.joint_names
        if self._same_shape(in_joint_state):
            # copy data:
            self.position.copy_(in_joint_state.position)
            self.velocity.copy_(in_joint_state.velocity)
            self.acceleration.copy_(in_joint_state.acceleration)
            # if self.jerk is not None:
            #    self.jerk.copy_(in_joint_state.jerk)
            return self
        else:
            log_info("Cloning JointState (breaks ref pointer)")
            # print(self.position.shape, in_joint_state.position.shape)
            # clone and create a new instance of JointState
            return in_joint_state.clone()

    def unsqueeze(self, idx: int):
        p = self.position.unsqueeze(idx)
        v = a = j = None
        if self.velocity is not None:
            v = self.velocity.unsqueeze(idx)
        if self.acceleration is not None:
            a = self.acceleration.unsqueeze(idx)
        if self.jerk is not None:
            j = self.jerk.unsqueeze(idx)
        return JointState(p, v, a, self.joint_names, jerk=j)

    def squeeze(self, dim: Optional[int] = 0):
        p = torch.squeeze(self.position, dim)
        v = a = j = None
        if self.velocity is not None:
            v = torch.squeeze(self.velocity, dim)
        if self.acceleration is not None:
            a = torch.squeeze(self.acceleration, dim)
        if self.jerk is not None:
            j = torch.squeeze(self.jerk, dim)
        return JointState(p, v, a, self.joint_names, jerk=j)

    def calculate_fd_from_position(self, dt: torch.Tensor):
        self.velocity = fd_tensor(self.position, dt)
        self.acceleration = fd_tensor(self.velocity, dt)
        self.jerk = fd_tensor(self.acceleration, dt)
        return self

    @staticmethod
    def zeros(
        size: Tuple[int], tensor_args: TensorDeviceType, joint_names: Optional[List[str]] = None
    ):
        return JointState(
            position=torch.zeros(size, device=tensor_args.device, dtype=tensor_args.dtype),
            velocity=torch.zeros(size, device=tensor_args.device, dtype=tensor_args.dtype),
            acceleration=torch.zeros(size, device=tensor_args.device, dtype=tensor_args.dtype),
            jerk=torch.zeros(size, device=tensor_args.device, dtype=tensor_args.dtype),
            joint_names=joint_names,
        )

    def detach(self):
        self.position = self.position.detach()
        self.velocity = self.velocity.detach()
        self.acceleration = self.acceleration.detach()
        if self.jerk is not None:
            self.jerk = self.jerk.detach()

        return self

    def get_ordered_joint_state(self, ordered_joint_names: List[str]) -> JointState:
        """Return joint state with a ordered joint names
        Args:
            ordered_joint_names (List[str]): _description_

        Returns:
            _type_: _description_
        """

        new_js = self.clone()
        new_js.inplace_reindex(ordered_joint_names)
        return new_js

    @profiler.record_function("joint_state/inplace_reindex")
    def inplace_reindex(self, joint_names: List[str]):
        if self.joint_names is None:
            raise ValueError("joint names are not specified in JointState")
        # get index of joint names:
        new_index_l = [self.joint_names.index(j) for j in joint_names]

        new_index = torch.as_tensor(new_index_l, device=self.position.device, dtype=torch.long)
        self.position = torch.index_select(self.position, -1, new_index)
        if self.velocity is not None:
            self.velocity = torch.index_select(self.velocity, -1, new_index)
        if self.acceleration is not None:
            self.acceleration = torch.index_select(self.acceleration, -1, new_index)
        if self.jerk is not None:
            self.jerk = torch.index_select(self.jerk, -1, new_index)
        self.joint_names = [self.joint_names[x] for x in new_index_l]

    def get_augmented_joint_state(
        self, joint_names, lock_joints: Optional[JointState] = None
    ) -> JointState:
        if lock_joints is None:
            return self.get_ordered_joint_state(joint_names)
        if joint_names is None or self.joint_names is None:
            raise ValueError("joint_names can't be None")

        # if some joints are locked, we assume that these joints are not in self.joint_names:
        if any(item in self.joint_names for item in lock_joints.joint_names):
            raise ValueError("lock_joints is also listed in self.joint_names")

        # append the lock_joints to existing joint state:
        new_js = self.clone()

        new_js = new_js.append_joints(lock_joints)

        new_js = new_js.get_ordered_joint_state(joint_names)
        return new_js

    def append_joints(self, js: JointState):
        if js.joint_names is None or len(js.joint_names) == 0:
            log_error("joint_names are required to append")

        current_shape = self.position.shape
        extra_len = len(js.joint_names)
        current_js = self
        one_dim = False
        # if joint state is of shape dof:
        if len(current_shape) == 1:
            current_js = self.unsqueeze(0)
            one_dim = True
            current_shape = current_js.position.shape

        if current_shape[:-1] != js.position.shape:
            if len(js.position.shape) > 1 and js.position.shape[0] > 1:
                raise ValueError(
                    "appending joints requires the new joints to have a shape matching current"
                    + " batch size or have a batch size of 1."
                )
        current_js.joint_names.extend(js.joint_names)

        if current_shape[:-1] == js.position.shape and len(current_shape) == len(js.position.shape):
            current_js.position = torch.cat((current_js.position, js.position), dim=-1)
            new_js = current_js
        else:
            current_shape = list(current_shape)
            current_shape[-1] += len(js.joint_names)
            new_js = JointState.zeros(
                current_shape, current_js.tensor_args, joint_names=self.joint_names
            )
            new_js.position[..., :-extra_len] = current_js.position
            if current_js.velocity is not None:
                new_js.velocity[..., :-extra_len] = current_js.velocity
            if current_js.acceleration is not None:
                new_js.acceleration[..., :-extra_len] = current_js.acceleration
            if current_js.jerk is not None:
                new_js.jerk[..., :-extra_len] = current_js.jerk

            new_js.position[..., -extra_len:] = js.position
            if js.velocity is not None:
                new_js.velocity[..., -extra_len:] = js.velocity
            if js.acceleration is not None:
                new_js.acceleration[..., -extra_len:] = js.acceleration
            if js.jerk is not None:
                new_js.jerk[..., -extra_len:] = js.jerk
        if one_dim:
            new_js = new_js.squeeze()
        return new_js

    def trim_trajectory(self, start_idx: int, end_idx: Optional[int] = None):
        if end_idx is None or end_idx == 0:
            end_idx = self.position.shape[-2]
        if len(self.position.shape) < 2:
            raise ValueError("JointState does not have horizon")
        pos = self.position[..., start_idx:end_idx, :].clone()
        vel = acc = jerk = None
        if self.velocity is not None:
            vel = self.velocity[..., start_idx:end_idx, :].clone()
        if self.acceleration is not None:
            acc = self.acceleration[..., start_idx:end_idx, :].clone()
        if self.jerk is not None:
            jerk = self.jerk[..., start_idx:end_idx, :].clone()
        return JointState(pos, vel, acc, self.joint_names, jerk, self.tensor_args)

    def scale(self, dt: Union[float, torch.Tensor]):
        vel = acc = jerk = None
        if self.velocity is not None:
            vel = self.velocity * dt
        if self.acceleration is not None:
            acc = self.acceleration * (dt**2)
        if self.jerk is not None:
            jerk = self.jerk * (dt**3)
        return JointState(self.position, vel, acc, self.joint_names, jerk, self.tensor_args)

    def scale_by_dt(self, dt: torch.Tensor, new_dt: torch.Tensor):
        vel, acc, jerk = jit_js_scale(self.velocity, self.acceleration, self.jerk, dt, new_dt)

        return JointState(self.position, vel, acc, self.joint_names, jerk, self.tensor_args)

    @property
    def shape(self):
        return self.position.shape

    def index_dof(self, idx: int):
        # get index of joint names:
        velocity = acceleration = jerk = None
        new_index = idx
        position = torch.index_select(self.position, -1, new_index)
        if self.velocity is not None:
            velocity = torch.index_select(self.velocity, -1, new_index)
        if self.acceleration is not None:
            acceleration = torch.index_select(self.acceleration, -1, new_index)
        if self.jerk is not None:
            jerk = torch.index_select(self.jerk, -1, new_index)
        joint_names = [self.joint_names[x] for x in new_index]
        return JointState(
            position=position,
            joint_names=joint_names,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
        )


@get_torch_jit_decorator()
def jit_js_scale(
    vel: Union[None, torch.Tensor],
    acc: Union[None, torch.Tensor],
    jerk: Union[None, torch.Tensor],
    dt: torch.Tensor,
    new_dt: torch.Tensor,
):
    scale_dt = dt / new_dt
    if vel is not None:
        vel = vel * scale_dt
    if acc is not None:
        acc = acc * scale_dt * scale_dt
    if jerk is not None:
        jerk = jerk * scale_dt * scale_dt * scale_dt
    return vel, acc, jerk


@get_torch_jit_decorator()
def jit_get_index(
    position: torch.Tensor,
    velocity: Union[torch.Tensor, None],
    acc: Union[torch.Tensor, None],
    jerk: Union[torch.Tensor, None],
    idx: torch.Tensor,
):

    position = position[idx]
    if velocity is not None:
        velocity = velocity[idx]
    if acc is not None:
        acc = acc[idx]
    if jerk is not None:
        jerk = jerk[idx]

    return position, velocity, acc, jerk


def fn_get_index(
    position: torch.Tensor,
    velocity: Union[torch.Tensor, None],
    acc: Union[torch.Tensor, None],
    jerk: Union[torch.Tensor, None],
    idx: torch.Tensor,
):

    position = position[idx]
    if velocity is not None:
        velocity = velocity[idx]
    if acc is not None:
        acc = acc[idx]
    if jerk is not None:
        jerk = jerk[idx]

    return position, velocity, acc, jerk


@get_torch_jit_decorator()
def jit_get_index_int(
    position: torch.Tensor,
    velocity: Union[torch.Tensor, None],
    acc: Union[torch.Tensor, None],
    jerk: Union[torch.Tensor, None],
    idx: int,
):

    position = position[idx]
    if velocity is not None:
        velocity = velocity[idx]
    if acc is not None:
        acc = acc[idx]
    if jerk is not None:
        jerk = jerk[idx]

    return position, velocity, acc, jerk
