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

"""This module contains aliases for structured Tensors, improving readability."""

# Third Party
import torch

# CuRobo
from curobo.util.logger import log_warn

T_DOF = torch.Tensor  #: Tensor of shape [degrees of freedom]
T_BDOF = torch.Tensor  #: Tensor of shape [batch, degrees of freedom]
T_BHDOF_float = torch.Tensor  #: Tensor of shape [batch, horizon, degrees of freedom]
T_HDOF_float = torch.Tensor  #: Tensor of shape [horizon, degrees of freedom]

T_BValue_float = torch.Tensor  #: Float Tensor of shape [batch, 1].
T_BHValue_float = torch.Tensor  #: Float Tensor of shape [batch, horizon, 1].
T_BValue_bool = torch.Tensor  #: Bool Tensor of shape [batch, horizon, 1].
T_BValue_int = torch.Tensor  #: Int Tensor of shape [batch, horizon, 1].

T_BPosition = torch.Tensor  #: Tensor of shape [batch, 3].
T_BQuaternion = torch.Tensor  #: Tensor of shape [batch, 4].
T_BRotation = torch.Tensor  #: Tensor of shape [batch, 3,3].
