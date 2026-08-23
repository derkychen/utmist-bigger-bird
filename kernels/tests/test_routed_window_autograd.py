"""Forward/backward parity tests for fused routed-window training attention."""

from __future__ import annotations

import pytest
import torch

from experiments.exp_18_confidence_gated.attention_core import _reference_routed_attention
from kernels import routed_window_attention_autograd, triton_available


@pytest.mark.skipif(not triton_available(), reason="Triton CUDA kernels not available")
def test_routed_window_autograd_parity():
    torch.manual_seed(0)
    bh, seq_len, head_dim = 2, 16, 16
    window = 4
    route = torch.tensor([[0, 3, 7], [1, 5, 9]], device="cuda")
    q = torch.randn(bh, seq_len, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(bh, seq_len, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(bh, seq_len, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)
    q_ref, k_ref, v_ref = (x.detach().float().requires_grad_(True) for x in (q, k, v))
    grad = torch.randn_like(q_ref)

    actual = routed_window_attention_autograd(q, k, v, route, window, causal=True)
    expected = _reference_routed_attention(
        q_ref, k_ref, v_ref, route, window, None, is_causal=True
    )
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)

    (actual.float() * grad).sum().backward()
    (expected * grad).sum().backward()
    torch.testing.assert_close(q.grad.float(), q_ref.grad, rtol=4e-2, atol=4e-2)
    torch.testing.assert_close(k.grad.float(), k_ref.grad, rtol=4e-2, atol=4e-2)
    torch.testing.assert_close(v.grad.float(), v_ref.grad, rtol=4e-2, atol=4e-2)
