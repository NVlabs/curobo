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

# CuRobo
from curobo.util.logger import log_warn

try:
    # Third Party
    from torchtyping import TensorType
except ImportError:
    log_warn("torchtyping could not be imported, falling back to basic types")
    TensorType = None
    # Third Party
    import torch
b_dof = ["batch", "dof"]
b_value = ["batch", "value"]
bh_value = ["batch", "horizon", "value"]
bh_dof = ["batch", "horizon", "dof"]
h_dof = ["horizon", "dof"]

if TensorType is not None:
    T_DOF = TensorType[tuple(["dof"] + [float])]
    T_BDOF = TensorType[tuple(b_dof + [float])]
    T_BValue_float = TensorType[tuple(b_value + [float])]
    T_BHValue_float = TensorType[tuple(bh_value + [float])]
    T_BValue_bool = TensorType[tuple(b_value + [bool])]
    T_BValue_int = TensorType[tuple(b_value + [int])]

    T_BPosition = TensorType["batch", "xyz":3, float]
    T_BQuaternion = TensorType["batch", "wxyz":4, float]
    T_BRotation = TensorType["batch", 3, 3, float]
    T_Position = TensorType["xyz":3, float]
    T_Quaternion = TensorType["wxyz":4, float]
    T_Rotation = TensorType[3, 3, float]

    T_BHDOF_float = TensorType[tuple(bh_dof + [float])]
    T_HDOF_float = TensorType[tuple(h_dof + [float])]
else:
    T_DOF = torch.Tensor
    T_BDOF = torch.Tensor
    T_BValue_float = torch.Tensor
    T_BHValue_float = torch.Tensor
    T_BValue_bool = torch.Tensor
    T_BValue_int = torch.Tensor

    T_BPosition = torch.Tensor
    T_BQuaternion = torch.Tensor
    T_BRotation = torch.Tensor
    T_Position = torch.Tensor
    T_Quaternion = torch.Tensor
    T_Rotation = torch.Tensor

    T_BHDOF_float = torch.Tensor
    T_HDOF_float = torch.Tensor
