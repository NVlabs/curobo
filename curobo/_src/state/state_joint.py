# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

# Standard Library
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

# Third Party
import numpy as np
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tensor import T_BDOF, T_DOF
from curobo._src.util.logging import deprecated, log_and_raise, log_info, log_warn
from curobo._src.util.tensor_util import (
    check_tensor_shapes,
    clone_if_not_none,
    copy_tensor,
)

from .filter_coeff import FilterCoeff
from .state_base import State
from .state_joint_jit_helpers import (
    clone_state_jit,
    fn_get_index,
    jit_get_index,
    jit_get_index_int,
    jit_joint_state_copy,
)

# Import ops functions
from .state_joint_ops import (
    append_joints_to_state,
    apply_kernel_to_joint_state,
    augment_joint_state,
    blend_joint_states,
    calculate_fd_from_position,
    cat_joint_states,
    joint_state_to_tensor,
    reindex_joint_state_inplace,
    reorder_joint_state,
    repeat_joint_state,
    repeat_joint_state_seeds,
    scale_joint_state,
    scale_joint_state_by_dt,
    scale_joint_state_time,
    stack_joint_states,
)
from .state_joint_trajectory_ops import (
    copy_joint_state_at_batch_seed_indices,
    copy_joint_state_at_index,
    copy_joint_state_only_index,
    gather_joint_state_by_seed,
    get_joint_state_at_horizon_index,
    index_joint_state_dof,
    trim_joint_state_trajectory,
)


@dataclass
class JointState(State):
    """Joint-space robot state (position and optional derivatives).

    Convention: use ``joint_state`` or ``js`` for JointState objects; use ``q`` for raw position tensors.
    """

    position: Union[List[float], T_DOF]
    velocity: Union[List[float], T_DOF, None] = None
    acceleration: Union[List[float], T_DOF, None] = None
    joint_names: Optional[List[str]] = None
    jerk: Union[List[float], T_DOF, None] = None  # Optional
    device_cfg: DeviceCfg = DeviceCfg()
    dt: Optional[torch.Tensor] = None
    aux_data: dict = field(default_factory=lambda: {})
    knot: Optional[torch.Tensor] = None
    knot_dt: Optional[torch.Tensor] = None
    control_space: Optional[ControlSpace] = None

    def __post_init__(self):
        if isinstance(self.position, torch.Tensor):
            self.device_cfg = DeviceCfg(self.position.device)

    def data_ptr(self):
        return self.position.data_ptr()

    @property
    def device(self) -> torch.device:
        return self.position.device

    @property
    def dtype(self) -> torch.dtype:
        return self.position.dtype

    @property
    def shape(self) -> torch.Size:
        return self.position.shape

    @property
    def ndim(self) -> int:
        return self.position.ndim

    @staticmethod
    def from_numpy(
        joint_names: List[str],
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        acceleration: Optional[np.ndarray] = None,
        jerk: Optional[np.ndarray] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
    ):
        pos = device_cfg.to_device(position)
        vel = acc = je = None
        if velocity is not None:
            vel = device_cfg.to_device(velocity)
        else:
            vel = pos * 0.0
        if acceleration is not None:
            acc = device_cfg.to_device(acceleration)
        else:
            acc = pos * 0.0
        if jerk is not None:
            je = device_cfg.to_device(jerk)
        else:
            je = pos * 0.0
        return JointState(pos, vel, acc, joint_names=joint_names, jerk=je, device_cfg=device_cfg)

    @staticmethod
    def from_position(position: T_BDOF, joint_names: Optional[List[str]] = None):
        return JointState(
            position=position,
            velocity=position * 0.0,
            acceleration=position * 0.0,
            jerk=position * 0.0,
            joint_names=joint_names,
        )

    @staticmethod
    def from_state_tensor(state_tensor, joint_names=None, dof=7):
        return JointState(
            state_tensor[..., :dof].contiguous(),
            state_tensor[..., dof : 2 * dof].contiguous(),
            state_tensor[..., 2 * dof : 3 * dof].contiguous(),
            jerk=state_tensor[..., 3 * dof : 4 * dof].contiguous(),
            joint_names=joint_names,
        )

    @staticmethod
    def from_list(position, velocity, acceleration, device_cfg: DeviceCfg()):
        joint_state = JointState(position, velocity, acceleration)
        joint_state = joint_state.to(device_cfg)
        return joint_state

    @staticmethod
    def zeros(
        size: Tuple[int], device_cfg: DeviceCfg, joint_names: Optional[List[str]] = None
    ):
        return JointState(
            position=torch.zeros(size, device=device_cfg.device, dtype=device_cfg.dtype),
            velocity=torch.zeros(size, device=device_cfg.device, dtype=device_cfg.dtype),
            acceleration=torch.zeros(size, device=device_cfg.device, dtype=device_cfg.dtype),
            jerk=torch.zeros(size, device=device_cfg.device, dtype=device_cfg.dtype),
            joint_names=joint_names,
            dt=torch.ones((size[0]), device=device_cfg.device, dtype=device_cfg.dtype),
        )

    def to(self, device_cfg: DeviceCfg):
        position = device_cfg.to_device(self.position)
        velocity = acceleration = jerk = None
        if self.velocity is not None:
            velocity = device_cfg.to_device(self.velocity)
        if self.acceleration is not None:
            acceleration = device_cfg.to_device(self.acceleration)
        if self.jerk is not None:
            jerk = device_cfg.to_device(self.jerk)
        return JointState(
            position,
            velocity,
            acceleration,
            jerk=jerk,
            dt=device_cfg.to_device(self.dt) if self.dt is not None else None,
            device_cfg=device_cfg,
            joint_names=self.joint_names,
        )

    @profiler.record_function("JointState/clone")
    def clone(self):
        j_names = None
        if self.joint_names is not None:
            j_names = self.joint_names.copy()

        position, velocity, acceleration, jerk, dt = clone_state_jit(
            self.position, self.velocity, self.acceleration, self.jerk, self.dt
        )
        return JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            dt=dt,
            joint_names=j_names,
            device_cfg=self.device_cfg,
            knot=clone_if_not_none(self.knot),
            knot_dt=clone_if_not_none(self.knot_dt),
            control_space=self.control_space,
        )

    def detach(self):
        self.position = self.position.detach()
        self.velocity = self.velocity.detach()
        self.acceleration = self.acceleration.detach()
        if self.jerk is not None:
            self.jerk = self.jerk.detach()
        if self.dt is not None:
            self.dt = self.dt.detach()
        return self

    def copy_reference(self, in_joint_state: JointState):
        """Copy reference to in_joint_state"""
        self.position = in_joint_state.position
        self.velocity = in_joint_state.velocity
        self.acceleration = in_joint_state.acceleration
        self.jerk = in_joint_state.jerk
        self.dt = in_joint_state.dt
        self.joint_names = in_joint_state.joint_names
        self.knot = in_joint_state.knot
        self.knot_dt = in_joint_state.knot_dt
        return self

    def _same_shape(self, new_js: JointState):
        same_shape = False
        same_shape = check_tensor_shapes(new_js.position, self.position)
        if not same_shape:
            return False
        if new_js.velocity is not None:
            same_shape = check_tensor_shapes(new_js.velocity, self.velocity)
            if not same_shape:
                return False
        if new_js.acceleration is not None:
            same_shape = check_tensor_shapes(new_js.acceleration, self.acceleration)
            if not same_shape:
                return False
        return same_shape

    @profiler.record_function("JointState/copy_")
    def copy_(self, in_joint_state: JointState, allow_clone: bool = True):
        if in_joint_state.joint_names is not None:
            self.joint_names = in_joint_state.joint_names

        if self._same_shape(in_joint_state):
            self.position, self.velocity, self.acceleration, self.jerk, self.dt = (
                jit_joint_state_copy(
                    self.position,
                    self.velocity,
                    self.acceleration,
                    self.jerk,
                    self.dt,
                    in_joint_state.position,
                    in_joint_state.velocity,
                    in_joint_state.acceleration,
                    in_joint_state.jerk,
                    in_joint_state.dt,
                )
            )
            return self
        else:
            if not allow_clone:
                log_and_raise(
                    "current state has shape: "
                    + str(self.position.shape)
                    + " while new shape is "
                    + str(in_joint_state.position.shape)
                )

            log_info("Cloning JointState (breaks ref pointer)")
            new_state = in_joint_state.clone()
            self.copy_reference(new_state)
            return self

    def copy_data(self, in_joint_state: JointState):
        """Copy data from in_joint_state to self"""
        log_warn("JointState.copy_data is deprecated, use JointState.copy_ instead")
        if not copy_tensor(in_joint_state.position, self.position):
            self.position = in_joint_state.position
            log_info("Cloning JointState")
        if not copy_tensor(in_joint_state.velocity, self.velocity):
            self.velocity = in_joint_state.velocity
        if not copy_tensor(in_joint_state.acceleration, self.acceleration):
            self.acceleration = in_joint_state.acceleration
        if not copy_tensor(in_joint_state.jerk, self.jerk):
            self.jerk = in_joint_state.jerk
        if not copy_tensor(in_joint_state.dt, self.dt):
            self.dt = in_joint_state.dt
        return self

    @profiler.record_function("JointState/unsqueeze")
    def unsqueeze(self, idx: int):
        p = self.position.unsqueeze(idx)
        v = a = j = knot = None
        if self.velocity is not None:
            v = self.velocity.unsqueeze(idx)
        if self.acceleration is not None:
            a = self.acceleration.unsqueeze(idx)
        if self.jerk is not None:
            j = self.jerk.unsqueeze(idx)
        if self.knot is not None:
            knot = self.knot.unsqueeze(idx)
        return JointState(
            p,
            v,
            a,
            self.joint_names,
            jerk=j,
            knot=knot,
            knot_dt=self.knot_dt,
            dt=self.dt,
            control_space=self.control_space,
        )

    def squeeze(self, dim: Optional[int] = 0):
        p = torch.squeeze(self.position, dim)
        v = a = j = None
        if self.velocity is not None:
            v = torch.squeeze(self.velocity, dim)
        if self.acceleration is not None:
            a = torch.squeeze(self.acceleration, dim)
        if self.jerk is not None:
            j = torch.squeeze(self.jerk, dim)
        return JointState(
            p,
            v,
            a,
            self.joint_names,
            jerk=j,
            knot=None if self.knot is None else torch.squeeze(self.knot, dim),
            knot_dt=self.knot_dt,
            control_space=self.control_space,
            dt=self.dt,
        )

    def view(self, *shape):
        return JointState(
            position=self.position.view(*shape) if self.position is not None else None,
            velocity=self.velocity.view(*shape) if self.velocity is not None else None,
            acceleration=self.acceleration.view(*shape) if self.acceleration is not None else None,
            jerk=self.jerk.view(*shape) if self.jerk is not None else None,
            joint_names=self.joint_names,
            dt=self.dt.view(*shape[:2]) if len(shape) > 2 and self.dt is not None else self.dt,
            device_cfg=self.device_cfg,
            knot=clone_if_not_none(self.knot),
            knot_dt=clone_if_not_none(self.knot_dt),
            control_space=self.control_space,
        )

    def __getitem__(self, idx):
        j = None
        v = a = None
        knot = None
        knot_dt = None
        max_idx = 0
        if isinstance(idx, List):
            idx = torch.as_tensor(idx, device=self.position.device, dtype=torch.long)
        if isinstance(idx, int):
            max_idx = idx
        elif isinstance(idx, torch.Tensor):
            max_idx = torch.max(idx)
        if max_idx >= self.position.shape[0]:
            log_and_raise(
                f"{max_idx} index out of range, current state is of length {self.position.shape}"
            )
        if isinstance(idx, int):
            p, v, a, j, dt = jit_get_index_int(
                self.position, self.velocity, self.acceleration, self.jerk, self.dt, idx
            )
        elif isinstance(idx, torch.Tensor):
            p, v, a, j, dt = jit_get_index(
                self.position, self.velocity, self.acceleration, self.jerk, self.dt, idx
            )
        else:
            p, v, a, j, dt = fn_get_index(
                self.position, self.velocity, self.acceleration, self.jerk, self.dt, idx
            )
        if self.knot is not None:
            knot = self.knot[idx]
        if self.knot_dt is not None:
            knot_dt = self.knot_dt
            if len(self.knot_dt.shape) > 0:
                knot_dt = self.knot_dt[idx]
        return JointState(
            p,
            v,
            a,
            joint_names=self.joint_names,
            jerk=j,
            dt=dt,
            knot=knot,
            knot_dt=knot_dt,
            control_space=self.control_space,
        )

    def __setitem__(self, idx: Union[int, torch.Tensor], value: JointState):
        if isinstance(idx, int):
            self.position[idx] = value.position
            self.velocity[idx] = value.velocity
            self.acceleration[idx] = value.acceleration
            self.jerk[idx] = value.jerk
            if self.dt is not None:
                self.dt[idx] = value.dt
        elif isinstance(idx, torch.Tensor):
            self.position[idx] = value.position
            self.velocity[idx] = value.velocity
            self.acceleration[idx] = value.acceleration
            self.jerk[idx] = value.jerk
            if self.dt is not None:
                self.dt[idx] = value.dt

    def __len__(self):
        return self.position.shape[0]

    # ========== Deprecated methods - use standalone functions instead ==========

    @deprecated("Use blend_joint_states() from state_joint_ops instead.")
    def blend(self, coeff: FilterCoeff, new_state: JointState):
        return blend_joint_states(self, new_state, coeff)

    @deprecated("Use joint_state_to_tensor() from state_joint_ops instead.")
    def get_state_tensor(self):
        return joint_state_to_tensor(self)

    @deprecated("Use stack_joint_states() from state_joint_ops instead.")
    def stack(self, new_state: JointState):
        return stack_joint_states(self, new_state)

    @deprecated("Use cat_joint_states() from state_joint_ops instead.")
    def cat(self, other_js: JointState, dim: int):
        return cat_joint_states(self, other_js, dim)

    @deprecated("Use repeat_joint_state() from state_joint_ops instead.")
    def repeat(self, repeat_input: List[int]):
        return repeat_joint_state(self, repeat_input)

    @deprecated("Use repeat_joint_state_seeds() from state_joint_ops instead.")
    def repeat_seeds(self, num_seeds: int):
        return repeat_joint_state_seeds(self, num_seeds)

    @deprecated("Use apply_kernel_to_joint_state() from state_joint_ops instead.")
    def apply_kernel(self, kernel_mat):
        return apply_kernel_to_joint_state(self, kernel_mat)

    @deprecated("Use scale_joint_state() from state_joint_ops instead.")
    def scale(self, dt: Union[float, torch.Tensor]):
        return scale_joint_state(self, dt)

    @deprecated("Use scale_joint_state_by_dt() from state_joint_ops instead.")
    def scale_by_dt(self, dt: torch.Tensor, new_dt: torch.Tensor):
        return scale_joint_state_by_dt(self, dt, new_dt)

    @deprecated("Use scale_joint_state_time() from state_joint_ops instead.")
    def scale_time(self, new_dt: torch.Tensor):
        return scale_joint_state_time(self, new_dt)

    @deprecated("Use calculate_fd_from_position() from state_joint_ops instead.")
    def calculate_fd_from_position(self, dt: Optional[torch.Tensor] = None):
        return calculate_fd_from_position(self, dt)

    def reorder(self, joint_names: List[str]) -> JointState:
        """Reorder joint state to match the given joint names order.

        Args:
            joint_names: List of joint names in the desired order.

        Returns:
            New JointState with joints reordered to match joint_names.
        """
        return reorder_joint_state(self, joint_names)

    def reindex(self, joint_names: List[str]):
        """Reindex joint state in-place to match the given joint names order.

        Args:
            joint_names: List of joint names in the desired order.
        """
        reindex_joint_state_inplace(self, joint_names)

    @deprecated("Use augment_joint_state() from state_joint_ops instead.")
    def get_augmented_joint_state(
        self, joint_names, lock_joints: Optional[JointState] = None
    ) -> JointState:
        return augment_joint_state(self, joint_names, lock_joints)

    @deprecated("Use append_joints_to_state() from state_joint_ops instead.")
    def append_joints(self, joint_state: JointState):
        return append_joints_to_state(self, joint_state)

    @deprecated("Use gather_joint_state_by_seed() from state_joint_trajectory_ops instead.")
    def gather_by_seed_index(self, idx: torch.Tensor):
        return gather_joint_state_by_seed(self, idx)

    @deprecated("Use copy_joint_state_only_index() from state_joint_trajectory_ops instead.")
    def copy_only_index(self, in_joint_state: JointState, idx: Union[int, torch.Tensor]):
        return copy_joint_state_only_index(self, in_joint_state, idx)

    @deprecated("Use copy_joint_state_at_index() from state_joint_trajectory_ops instead.")
    def copy_at_index(self, in_joint_state: JointState, idx: Union[int, torch.Tensor]):
        copy_joint_state_at_index(self, in_joint_state, idx)

    @deprecated("Use copy_joint_state_at_batch_seed_indices() from state_joint_trajectory_ops.")
    def copy_at_batch_seed_indices(
        self, in_joint_state: JointState, batch_idx: torch.Tensor, seed_idx: torch.Tensor
    ):
        return copy_joint_state_at_batch_seed_indices(self, in_joint_state, batch_idx, seed_idx)

    @deprecated("Use get_joint_state_at_horizon_index() from state_joint_trajectory_ops instead.")
    def get_trajectory_at_horizon_index(self, horizon_index: int):
        return get_joint_state_at_horizon_index(self, horizon_index)

    @deprecated("Use trim_joint_state_trajectory() from state_joint_trajectory_ops instead.")
    def trim_trajectory(self, start_idx: int, end_idx: Optional[int] = None):
        return trim_joint_state_trajectory(self, start_idx, end_idx)

    @deprecated("Use index_joint_state_dof() from state_joint_trajectory_ops instead.")
    def index_dof(self, idx: int):
        return index_joint_state_dof(self, idx)
