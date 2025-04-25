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

"""Optimization module containing several numerical solvers.

Base for an opimization solver is at :class:`opt_base.Optimizer`. cuRobo provides two base classes
for implementing two popular ways to optimize, (1) using particles
with :class:`particle.particle_opt_base.ParticleOptBase` and (2) using Newton/Quasi-Newton solvers
with :class:`newton.newton_base.NewtonOptBase`. :class:`newton.newton_base.NewtonOptBase` contains
implementations of several line search schemes. Note that these line search schemes are approximate
as cuRobo tries different line search magnitudes in parallel and chooses the largest that satisfies
line search conditions.

"""
