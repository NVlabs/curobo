# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""This module contains aliases for structured Tensors, improving readability."""

# Third Party
import torch

T_DOF = torch.Tensor  #: Tensor of shape [degrees of freedom]
T_BDOF = torch.Tensor  #: Tensor of shape [batch, degrees of freedom]
T_BHDOF_float = torch.Tensor  #: Tensor of shape [batch, horizon, degrees of freedom]
T_HDOF_float = torch.Tensor  #: Tensor of shape [horizon, degrees of freedom]

T_BHValue_bool = torch.Tensor  #: Float Tensor of shape [batch, horizon, 1].
T_BValue_float = torch.Tensor  #: Float Tensor of shape [batch, 1].
T_BHValue_float = torch.Tensor  #: Float Tensor of shape [batch, horizon, 1].
T_BValue_bool = torch.Tensor  #: Bool Tensor of shape [batch, horizon, 1].
T_BValue_int = torch.Tensor  #: Int Tensor of shape [batch, horizon, 1].

T_BPosition = torch.Tensor  #: Tensor of shape [batch, 3].
T_BQuaternion = torch.Tensor  #: Tensor of shape [batch, 4].
T_BRotation = torch.Tensor  #: Tensor of shape [batch, 3,3].
