"""Parity tests for sparse_gather_attention vs PyTorch reference."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from shared.kernels import triton_available
from shared.kernels.gather import sparse_gather_attention


def _ref_gather_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    token_idx: torch.Tensor,
    key_mask: torch.Tensor | None,
    *,
    scale: float,
) -> torch.Tensor:
    bh, tgt_len, dim = q.shape
    m = token_idx.size(-1)
    idx_gather = token_idx.unsqueeze(-1).expand(-1, -1, -1, dim)
    k_sel = torch.gather(k.unsqueeze(1).expand(bh, tgt_len, k.size(1), dim), 2, idx_gather)
    v_sel = torch.gather(v.unsqueeze(1).expand(bh, tgt_len, v.size(1), dim), 2, idx_gather)
    scores = torch.matmul(q.unsqueeze(2), k_sel.transpose(-1, -2)).squeeze(2) * scale
    if key_mask is not None:
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
    attn = F.softmax(scores, dim=-1)
    return torch.bmm(
        attn.reshape(bh * tgt_len, 1, m),
        v_sel.reshape(bh * tgt_len, m, dim),
    ).reshape(bh, tgt_len, dim)


@pytest.mark.skipif(not triton_available(), reason="Triton CUDA kernels not available")
def test_sparse_gather_attention_parity():
    torch.manual_seed(0)
    bh, tgt_len, head_dim, src_len, m = 4, 8, 32, 24, 6
    q = torch.randn(bh, tgt_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(bh, src_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(bh, src_len, head_dim, device="cuda", dtype=torch.float16)
    token_idx = torch.randint(0, src_len, (bh, tgt_len, m), device="cuda", dtype=torch.int32)
    key_mask = torch.ones(bh, tgt_len, m, device="cuda", dtype=torch.bool)
    key_mask[:, :, -1] = False

    scale = head_dim ** -0.5
    out_triton = sparse_gather_attention(q, k, v, token_idx, key_mask, scale=scale)
    out_ref = _ref_gather_attention(
        q.float(), k.float(), v.float(), token_idx, key_mask, scale=scale
    ).to(torch.float16)

    torch.testing.assert_close(out_triton, out_ref, rtol=1e-2, atol=1e-2)
