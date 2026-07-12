"""Parity tests for elu_linear_attention vs PyTorch reference (exp_2 global branch)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from shared.kernels import triton_available
from shared.kernels.linear import LINEAR_EPS, elu_linear_attention


def _ref_elu_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Mirror LightningHybridAttention global path (ELU linear attention)."""
    q_l = F.elu(q) + 1.0
    k_l = F.elu(k) + 1.0
    if key_mask is not None:
        k_l = k_l * key_mask.unsqueeze(-1)
    kv = torch.bmm(k_l.transpose(1, 2), v)
    z = k_l.sum(dim=1, keepdim=True)
    num = torch.bmm(q_l, kv)
    den = torch.bmm(q_l, z.transpose(1, 2))
    return num / (den + LINEAR_EPS)


@pytest.mark.skipif(not triton_available(), reason="Triton CUDA kernels not available")
def test_elu_linear_attention_parity():
    torch.manual_seed(0)
    bh, tgt_len, head_dim, src_len = 4, 16, 32, 20
    q = torch.randn(bh, tgt_len, head_dim, device="cuda", dtype=torch.float32)
    k = torch.randn(bh, src_len, head_dim, device="cuda", dtype=torch.float32)
    v = torch.randn(bh, src_len, head_dim, device="cuda", dtype=torch.float32)

    out_triton = elu_linear_attention(q, k, v, None)
    out_ref = _ref_elu_linear_attention(q, k, v, None)
    torch.testing.assert_close(out_triton, out_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not triton_available(), reason="Triton CUDA kernels not available")
def test_elu_linear_attention_parity_with_mask():
    torch.manual_seed(1)
    bh, tgt_len, head_dim, src_len = 4, 12, 32, 18
    q = torch.randn(bh, tgt_len, head_dim, device="cuda", dtype=torch.float32)
    k = torch.randn(bh, src_len, head_dim, device="cuda", dtype=torch.float32)
    v = torch.randn(bh, src_len, head_dim, device="cuda", dtype=torch.float32)
    key_mask = torch.ones(bh, src_len, device="cuda", dtype=torch.bool)
    key_mask[:, -3:] = False

    out_triton = elu_linear_attention(q, k, v, key_mask)
    out_ref = _ref_elu_linear_attention(q, k, v, key_mask)
    torch.testing.assert_close(out_triton, out_ref, rtol=1e-3, atol=1e-3)
