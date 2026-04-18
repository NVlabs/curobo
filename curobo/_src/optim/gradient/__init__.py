# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""This module contains Newton/Quasi-Newton solvers.

Gradient descent:
1. compute cost and gradient (delta x)
2. store best x based on cost
3. use fixed step size and get updated x
4. repeat until convergence (1-3)

GD with line search:
1. compute cost and gradient (delta x) for different step sizes.
2. find best step size using line search conditions.
3. store best x based on best step size.
4. repeat until convergence (1-3)


LBFGS with line search:
1. new_x = x + step_size * step_direction
2. compute cost, gradient (delta x) for different step sizes at x.
3. find best step size using line search conditions.
4. store best x based on best step size.
5. compute step direction given gradient
6. repeat until convergence (1-5)



"""
