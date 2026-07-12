"""Parity tests for band_attention vs PyTorch reference (exp_2 local branch)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from shared.kernels import triton_available
from shared.kernels.band import band_attention


def _ref_band_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    radius: int,
    key_mask: torch.Tensor | None,
    *,
    scale: float,
) -> torch.Tensor:
    """Mirror LightningHybridAttention local path (symmetric band softmax)."""
    bh, tgt_len, _ = q.shape
    src_len = k.size(1)
    scores = torch.bmm(q, k.transpose(1, 2)) * scale
    idx_q = torch.arange(tgt_len, device=q.device).unsqueeze(1)
    idx_k = torch.arange(src_len, device=k.device).unsqueeze(0)
    local_mask = (torch.abs(idx_q - idx_k) <= radius).unsqueeze(0).expand(bh, -1, -1)
    if key_mask is not None:
        local_mask = local_mask & key_mask.unsqueeze(1)
    scores = scores.masked_fill(~local_mask, -1e9)
    attn = F.softmax(scores, dim=-1)
    return torch.bmm(attn, v)


@pytest.mark.skipif(not triton_available(), reason="Triton CUDA kernels not available")
def test_band_attention_parity():
    torch.manual_seed(0)
    bh, seq_len, head_dim = 4, 32, 32
    radius = 4
    q = torch.randn(bh, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(bh, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(bh, seq_len, head_dim, device="cuda", dtype=torch.float16)

    out_triton = band_attention(q, k, v, radius, None, scale=1.0)
    out_ref = _ref_band_attention(
        q.float(), k.float(), v.float(), radius, None, scale=1.0
    ).to(torch.float16)
    torch.testing.assert_close(out_triton, out_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not triton_available(), reason="Triton CUDA kernels not available")
def test_band_attention_parity_with_mask():
    torch.manual_seed(1)
    bh, seq_len, head_dim = 4, 32, 32
    radius = 6
    q = torch.randn(bh, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(bh, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(bh, seq_len, head_dim, device="cuda", dtype=torch.float16)
    key_mask = torch.ones(bh, seq_len, device="cuda", dtype=torch.bool)
    key_mask[:, -5:] = False  # padding tail

    out_triton = band_attention(q, k, v, radius, key_mask, scale=1.0)
    out_ref = _ref_band_attention(
        q.float(), k.float(), v.float(), radius, key_mask, scale=1.0
    ).to(torch.float16)
    # Padding query rows are excluded from comparison (their output is unused downstream).
    valid_rows = key_mask.unsqueeze(-1)
    torch.testing.assert_close(
        torch.where(valid_rows, out_triton, torch.zeros_like(out_triton)),
        torch.where(valid_rows, out_ref, torch.zeros_like(out_ref)),
        rtol=1e-2,
        atol=1e-2,
    )
