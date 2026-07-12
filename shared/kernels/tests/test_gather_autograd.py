"""Autograd parity tests for gather_attention_autograd (fwd + dQ/dK/dV)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from shared.kernels import triton_available
from shared.kernels.gather_autograd import gather_attention_autograd


def _ref_gather_attention(q, k, v, token_idx, key_mask, scale):
    """PyTorch reference: gather selected keys/values then dense softmax attention."""
    BH, T, D = q.shape
    bh = torch.arange(BH, device=q.device).view(BH, 1, 1)
    k_sel = k[bh, token_idx]  # [BH, T, M, D]
    v_sel = v[bh, token_idx]
    scores = (q.unsqueeze(2) * k_sel).sum(-1) * scale  # [BH, T, M]
    if key_mask is not None:
        scores = scores.masked_fill(~key_mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return (attn.unsqueeze(-1) * v_sel).sum(2)


def _make_inputs(BH, T, S, D, M, with_mask, seed):
    torch.manual_seed(seed)
    q = torch.randn(BH, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn(BH, S, D, device="cuda", dtype=torch.float32, requires_grad=True)
    v = torch.randn(BH, S, D, device="cuda", dtype=torch.float32, requires_grad=True)
    token_idx = torch.randint(0, S, (BH, T, M), device="cuda", dtype=torch.int64)
    key_mask = None
    if with_mask:
        key_mask = torch.rand(BH, T, M, device="cuda") > 0.25
        key_mask[..., 0] = True  # guarantee at least one valid key per row
    return q, k, v, token_idx, key_mask


@pytest.mark.skipif(not triton_available(), reason="Triton CUDA kernels not available")
@pytest.mark.parametrize("with_mask", [False, True])
def test_gather_autograd_parity(with_mask):
    BH, T, S, D, M = 6, 48, 128, 64, 16
    scale = D ** -0.5
    q, k, v, idx, mask = _make_inputs(BH, T, S, D, M, with_mask, seed=0)
    qr, kr, vr = (t.detach().clone().requires_grad_(True) for t in (q, k, v))

    out = gather_attention_autograd(q, k, v, idx, mask, scale=scale)
    out_ref = _ref_gather_attention(qr, kr, vr, idx, mask, scale)
    torch.testing.assert_close(out, out_ref, rtol=2e-3, atol=2e-3)

    g = torch.randn_like(out)
    out.backward(g)
    out_ref.backward(g)

    torch.testing.assert_close(q.grad, qr.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(k.grad, kr.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(v.grad, vr.grad, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not triton_available(), reason="Triton CUDA kernels not available")
def test_gather_autograd_duplicate_indices():
    """Many queries selecting the same key must accumulate dK/dV correctly (atomics)."""
    BH, T, S, D, M = 4, 32, 64, 32, 8
    scale = D ** -0.5
    torch.manual_seed(3)
    q = torch.randn(BH, T, D, device="cuda", requires_grad=True)
    k = torch.randn(BH, S, D, device="cuda", requires_grad=True)
    v = torch.randn(BH, S, D, device="cuda", requires_grad=True)
    # Heavy collisions: only 5 distinct key indices shared across all queries.
    token_idx = torch.randint(0, 5, (BH, T, M), device="cuda", dtype=torch.int64)
    qr, kr, vr = (t.detach().clone().requires_grad_(True) for t in (q, k, v))

    out = gather_attention_autograd(q, k, v, token_idx, None, scale=scale)
    out_ref = _ref_gather_attention(qr, kr, vr, token_idx, None, scale)
    g = torch.randn_like(out)
    out.backward(g)
    out_ref.backward(g)

    torch.testing.assert_close(q.grad, qr.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(k.grad, kr.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(v.grad, vr.grad, rtol=2e-3, atol=2e-3)
