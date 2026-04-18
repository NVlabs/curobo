# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Optimization module containing several numerical solvers.

Base for an opimization solver is at :class:`opt_base.Optimizer`. cuRobo provides two base classes
for implementing two popular ways to optimize, (1) using particles
with :class:`particle.particle_opt_base.ParticleOptBase` and (2) using Newton/Quasi-Newton solvers
with :class:`newton.newton_base.NewtonOptBase`. :class:`newton.newton_base.NewtonOptBase` contains
implementations of several line search schemes. Note that these line search schemes are approximate
as cuRobo tries different line search magnitudes in parallel and chooses the largest that satisfies
line search conditions.

"""
