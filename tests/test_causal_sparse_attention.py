"""Correctness tests for causal_sparse_attention and exp 13 routing."""

from __future__ import annotations

import torch

from sparse_attn_utils import causal_sparse_attention
from experiments.exp_13_dynamic_context.model_llama import dynamic_route_indices


def _reference(Q, K, V, routed, W, token_mask, num_heads, routed_mask=None, dedupe=False):
    """Explicit per-query key lists (duplicates kept unless deduped)."""
    BH, T, _ = Q.shape
    out = torch.zeros_like(Q)
    for bh in range(BH):
        b = bh // num_heads
        for q in range(T):
            keys = list(range(max(0, q - W + 1), q + 1))
            for j, idx in enumerate(routed[bh].tolist()):
                if routed_mask is not None and not routed_mask[bh, j]:
                    continue
                if idx > q or (dedupe and idx > q - W):
                    continue
                keys.append(idx)
            if token_mask is not None:
                keys = [key for key in keys if token_mask[b, key]]
            if not keys:
                continue
            keys = torch.tensor(keys)
            attn = torch.softmax(Q[bh, q] @ K[bh, keys].T, dim=-1)
            out[bh, q] = attn @ V[bh, keys]
    return out


def _inputs(B=2, H=2, T=90, d=8, k=12, seed=0):
    torch.manual_seed(seed)
    BH = B * H
    Q, K, V = (torch.randn(BH, T, d, dtype=torch.float64) for _ in range(3))
    routed = torch.randint(0, T, (BH, k))
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, :7] = False
    return Q, K, V, routed, mask, B, H


def _valid_rows(out, mask, H):
    return out[mask.repeat_interleave(H, dim=0)]


def test_matches_reference_with_padding_and_duplicates():
    Q, K, V, routed, mask, B, H = _inputs()
    out = causal_sparse_attention(Q, K, V, routed, local_window=16, token_mask=mask,
                                  bsz=B, num_heads=H, query_chunk=32)
    ref = _reference(Q, K, V, routed, 16, mask, H)
    torch.testing.assert_close(_valid_rows(out, mask, H), _valid_rows(ref, mask, H))


def test_dedupe_and_routed_mask():
    Q, K, V, routed, mask, B, H = _inputs(seed=1)
    routed_mask = torch.rand(routed.shape) > 0.3
    out = causal_sparse_attention(Q, K, V, routed, local_window=10, token_mask=mask,
                                  bsz=B, num_heads=H, query_chunk=25,
                                  routed_mask=routed_mask, dedupe_local=True)
    ref = _reference(Q, K, V, routed, 10, mask, H, routed_mask=routed_mask, dedupe=True)
    torch.testing.assert_close(_valid_rows(out, mask, H), _valid_rows(ref, mask, H))


def test_window_longer_than_sequence():
    Q, K, V, routed, _, _, _ = _inputs(B=1, H=1, T=40)
    out = causal_sparse_attention(Q, K, V, routed, local_window=64, bsz=1, num_heads=1,
                                  dedupe_local=True)
    scores = (Q @ K.transpose(1, 2)).masked_fill(torch.ones(40, 40).triu(1).bool(), float("-inf"))
    torch.testing.assert_close(out, torch.softmax(scores, dim=-1) @ V)


def test_exp13_routing_budget_and_sinks():
    torch.manual_seed(0)
    Q = torch.randn(4, 16, 32) / 32 ** 0.5
    K = torch.randn(4, 8192, 32)
    idx, routed_mask = dynamic_route_indices(
        Q, K, "length", min_budget=64, max_budget=1024, tokens_per_key=64, mass=0.95,
        num_route_queries=4, num_sinks=4,
    )
    assert idx.shape == (4, 128 + 4) and routed_mask is None
    assert all({0, 1, 2, 3} <= set(row.tolist()) for row in idx)

    K[0, 500] = Q[0, -1] * 40
    idx, routed_mask = dynamic_route_indices(
        Q, K, "mass", min_budget=16, max_budget=4096, tokens_per_key=64, mass=0.9,
        num_route_queries=1, num_sinks=4,
    )
    budgets = routed_mask.sum(dim=1)
    assert budgets[0] == 16 + 4 and budgets[1:].min() > budgets[0]
    assert 500 in idx[0, : budgets[0]].tolist()
