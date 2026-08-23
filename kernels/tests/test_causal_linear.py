"""Parity tests for the fused causal low-rank linear kernel."""

from __future__ import annotations

import pytest
import torch

from kernels import causal_linear_attention, triton_available


def _reference(qf, kf, v, mask=None):
    bh, t, feature_dim = qf.shape
    value_dim = v.size(-1)
    state = torch.zeros(bh, feature_dim, value_dim, device=qf.device, dtype=torch.float32)
    z = torch.zeros(bh, feature_dim, device=qf.device, dtype=torch.float32)
    out = []
    for pos in range(t):
        k = kf[:, pos].float()
        if mask is not None:
            k = k * mask[:, pos].unsqueeze(-1)
        state = state + k.unsqueeze(-1) * v[:, pos].float().unsqueeze(-2)
        z = z + k
        num = torch.einsum("bf,bfd->bd", qf[:, pos].float(), state)
        den = (qf[:, pos].float() * z).sum(dim=-1, keepdim=True)
        out.append(num / (den + 1e-6))
    return torch.stack(out, dim=1).to(v.dtype)


@pytest.mark.skipif(not triton_available(), reason="Triton CUDA kernels not available")
def test_causal_linear_attention_parity():
    torch.manual_seed(0)
    bh, seq_len, feature_dim, value_dim = 4, 32, 8, 16
    qf = (torch.rand(bh, seq_len, feature_dim, device="cuda", dtype=torch.float16) + 0.1)
    kf = (torch.rand(bh, seq_len, feature_dim, device="cuda", dtype=torch.float16) + 0.1)
    v = torch.randn(bh, seq_len, value_dim, device="cuda", dtype=torch.float16)
    actual = causal_linear_attention(qf, kf, v)
    expected = _reference(qf, kf, v)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not triton_available(), reason="Triton CUDA kernels not available")
def test_causal_linear_attention_mask_parity():
    torch.manual_seed(1)
    bh, seq_len, feature_dim, value_dim = 2, 24, 8, 16
    qf = (torch.rand(bh, seq_len, feature_dim, device="cuda", dtype=torch.float16) + 0.1)
    kf = (torch.rand(bh, seq_len, feature_dim, device="cuda", dtype=torch.float16) + 0.1)
    v = torch.randn(bh, seq_len, value_dim, device="cuda", dtype=torch.float16)
    mask = torch.ones(bh, seq_len, device="cuda", dtype=torch.bool)
    mask[:, :3] = False
    actual = causal_linear_attention(qf, kf, v, mask)
    expected = _reference(qf, kf, v, mask)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
