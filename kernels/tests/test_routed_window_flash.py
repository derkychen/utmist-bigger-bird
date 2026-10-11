"""Parity tests for the tiled causal local plus routed kernel."""

from __future__ import annotations

import pytest
import torch

from kernels import routed_window_attention_flash, triton_available
from sparse_attn_utils import causal_sparse_attention

pytestmark = pytest.mark.skipif(not triton_available(), reason="Triton CUDA kernels not available")


def _inputs(bh, seq_len, head_dim, n_routed, duplicates, seed):
    torch.manual_seed(seed)
    q, k, v = (
        torch.randn(bh, seq_len, head_dim, device="cuda", dtype=torch.float16) * 0.5
        for _ in range(3)
    )
    if duplicates:
        routed = torch.randint(0, seq_len, (bh, n_routed), device="cuda")
    else:
        routed = torch.stack([torch.randperm(seq_len, device="cuda")[:n_routed] for _ in range(bh)])
    return q, k, v, routed


@pytest.mark.parametrize(
    "seq_len,head_dim,window,n_routed,dedupe,duplicates",
    [
        (300, 64, 64, 50, False, False),
        (257, 128, 256, 70, True, True),
        (90, 128, 256, 20, True, False),
        (200, 32, 17, 64, False, True),
    ],
)
def test_matches_torch_path(seq_len, head_dim, window, n_routed, dedupe, duplicates):
    q, k, v, routed = _inputs(4, seq_len, head_dim, n_routed, duplicates, seed=seq_len)
    expected = causal_sparse_attention(
        q.double(), k.double(), v.double(), routed, local_window=window,
        bsz=2, num_heads=2, dedupe_local=dedupe,
    )
    actual = routed_window_attention_flash(q, k, v, routed, window, num_heads=2, dedupe_local=dedupe)
    torch.testing.assert_close(actual.double(), expected, rtol=0, atol=2e-3)


def test_padding_and_routed_mask():
    batch, heads = 2, 2
    q, k, v, routed = _inputs(batch * heads, 333, 64, 80, True, seed=7)
    token_mask = torch.ones(batch, 333, dtype=torch.bool, device="cuda")
    token_mask[0, :21] = False
    routed_mask = torch.rand(routed.shape, device="cuda") > 0.3

    expected = causal_sparse_attention(
        q.double(), k.double(), v.double(), routed, local_window=40,
        token_mask=token_mask, bsz=batch, num_heads=heads,
        routed_mask=routed_mask, dedupe_local=True,
    )
    actual = routed_window_attention_flash(
        q, k, v, routed, 40, token_mask=token_mask, num_heads=heads,
        routed_mask=routed_mask, dedupe_local=True,
    )
    valid = token_mask.repeat_interleave(heads, dim=0)
    torch.testing.assert_close(actual[valid].double(), expected[valid], rtol=0, atol=2e-3)


def test_unexpanded_gqa_kv():
    q, _, _, routed = _inputs(8, 400, 64, 60, False, seed=3)
    k, v = (torch.randn(2, 400, 64, device="cuda", dtype=torch.float16) * 0.5 for _ in range(2))
    expected = causal_sparse_attention(
        q.double(), k.repeat_interleave(4, 0).double(), v.repeat_interleave(4, 0).double(),
        routed, local_window=64, bsz=1, num_heads=8,
    )
    actual = routed_window_attention_flash(q, k, v, routed, 64, num_heads=8)
    torch.testing.assert_close(actual.double(), expected, rtol=0, atol=2e-3)
