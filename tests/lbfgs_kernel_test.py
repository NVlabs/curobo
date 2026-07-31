# Third Party
import pytest
import torch

# CuRobo
from curobo.curobolib.opt import lbfgs_step_cu


def get_lbfgs_reference(rho, y, s, q, grad_q, x_0, grad_0):
    y_new = grad_q - grad_0
    s_new = q - x_0
    numerator = torch.sum(y_new * s_new, dim=-1)

    y_history = torch.cat((y[1:, :, :, 0], y_new.unsqueeze(0)))
    s_history = torch.cat((s[1:, :, :, 0], s_new.unsqueeze(0)))
    rho_history = torch.cat((rho[1:, :, 0, 0], numerator.reciprocal().unsqueeze(0)))

    value = grad_q.clone()
    alpha = torch.empty_like(rho_history)
    for i in range(y_history.shape[0] - 1, -1, -1):
        alpha[i] = torch.sum(value * s_history[i], dim=-1) * rho_history[i]
        value -= alpha[i].unsqueeze(-1) * y_history[i]

    denominator = torch.sum(y_new.square(), dim=-1)
    value *= torch.clamp_min(numerator / denominator, 0.0).unsqueeze(-1)

    for i in range(y_history.shape[0]):
        beta = torch.sum(value * y_history[i], dim=-1) * rho_history[i]
        value += (alpha[i] - beta).unsqueeze(-1) * s_history[i]

    return -value


@pytest.mark.parametrize("v_dim", [7, 31, 33, 175, 224])
@pytest.mark.parametrize("use_shared_buffers", [False, True])
def test_lbfgs_fused_kernel_handles_warp_boundaries(v_dim, use_shared_buffers):
    # Cover partial warps, a partial final warp, and a seven-warp reduction.
    history = 6
    batch = 4
    device = torch.device("cuda")

    torch.manual_seed(7)
    step = torch.zeros((batch, v_dim), device=device, dtype=torch.float32)
    rho = torch.full((history, batch, 1, 1), 0.01, device=device, dtype=torch.float32)
    y = 0.05 * torch.randn((history, batch, v_dim, 1), device=device)
    s = 0.05 * torch.randn((history, batch, v_dim, 1), device=device)
    q = torch.randn((batch, v_dim), device=device, dtype=torch.float32)
    grad_q = torch.randn((batch, v_dim), device=device, dtype=torch.float32)
    delta = torch.linspace(0.01, 0.2, v_dim, device=device).expand(batch, -1)
    x_0 = q - delta
    grad_0 = grad_q - delta.flip(-1)

    expected = get_lbfgs_reference(
        rho.clone(),
        y.clone(),
        s.clone(),
        q.clone(),
        grad_q.clone(),
        x_0.clone(),
        grad_0.clone(),
    )

    lbfgs_step_cu.forward(
        step,
        rho,
        y,
        s,
        q,
        grad_q,
        x_0,
        grad_0,
        0.01,
        batch,
        history,
        v_dim,
        True,
        use_shared_buffers,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(step, expected, rtol=5e-4, atol=5e-5)
