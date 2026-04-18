# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
def ceil_div(a: int, b: int) -> int:
    """Ceiling division helper"""
    return (a + b - 1) // b
