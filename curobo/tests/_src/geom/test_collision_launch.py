# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check capped collision blocks with partial warps and strided sphere queries."""

import pytest
import torch

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.warp import init_warp


@pytest.mark.parametrize("num_spheres", [1, 33, 65, 129, 513, 1025])
@pytest.mark.parametrize("swept", [False, True])
@pytest.mark.parametrize("summed", [False, True])
def test_collision_reduction_and_graph_replay(
    num_spheres: int, swept: bool, summed: bool, cuda_device_cfg: DeviceCfg
) -> None:
    """Preserve analytic costs and gradients across all sphere passes and graph replay."""
    init_warp(quiet=True, device_cfg=cuda_device_cfg)
    obstacle = Cuboid(name="box", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.4, 0.4, 0.4])
    scene = SceneCollision.from_config(
        SceneCollisionCfg(device_cfg=cuda_device_cfg, scene_model=SceneCfg(cuboid=[obstacle]))
    )
    query = torch.tensor([0.25, 0.03, 0.01, 0.1], device=cuda_device_cfg.device)
    query = query.repeat(2, 3, num_spheres, 1)
    query[:, :, 1::5, 3] = -1.0
    query.requires_grad_()
    masks = torch.ones((1, num_spheres), dtype=torch.uint32, device=query.device)
    masks[:, 2::3] = 0
    active = (query.detach()[..., 3] > 0) & (masks.view(torch.int32) != 0)
    expected = active.to(torch.float32) * 0.05
    if summed:
        expected = expected.sum(dim=-1, keepdim=True)
    args = dict(
        query_spheres=query,
        collision_buffer=CollisionBuffer.from_shape(query.shape, cuda_device_cfg),
        weight=torch.ones(1, device=query.device),
        activation_distance=torch.zeros(1, device=query.device),
        sphere_collision_mask=masks,
        sum_distance=summed,
    )
    if swept:
        args["trajectory_dt"] = torch.full((2,), 0.1, device=query.device)
        run = scene.get_swept_sphere_distance_raw
    else:
        run = scene.get_sphere_distance_raw
    run(**args)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(**args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = run(**args)
    graph.replay()
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
    result.sum().backward()
    expected_gradient = torch.zeros_like(query)
    expected_gradient[..., 0] = -active.to(torch.float32)
    torch.testing.assert_close(query.grad, expected_gradient, atol=1e-6, rtol=1e-6)
    with torch.no_grad():
        query[..., 0].add_(1.0)
    graph.replay()
    torch.testing.assert_close(result, torch.zeros_like(result))
