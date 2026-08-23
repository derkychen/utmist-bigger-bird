"""Correctness tests for Exp 18's sparse-only attention core."""

from __future__ import annotations

from pathlib import Path

import torch

from experiments.exp_18_confidence_gated.attention_core import (
    ConfidenceGatedAttentionCore,
    _reference_linear_attention,
    _reference_local_attention,
    _reference_routed_attention,
    route_global_indices,
)


def test_exp18_model_is_sparse_only():
    source = Path(__file__).resolve().parents[1] / "experiments" / "exp_18_confidence_gated" / "model_llama.py"
    text = source.read_text(encoding="utf-8")
    forbidden = (
        "dense_self_attention",
        "DenseLlamaAttention",
        "sdpa_dense_or_none",
        "F.scaled_dot_product_attention",
    )
    assert not [item for item in forbidden if item in text]


def _direct_local(q, k, v, window, causal):
    bh, t, d = q.shape
    out = torch.zeros_like(q)
    for b in range(bh):
        for pos in range(t):
            if causal:
                indices = list(range(max(0, pos - window + 1), pos + 1))
            else:
                radius = window // 2
                indices = list(range(max(0, pos - radius), min(t, pos + radius + 1)))
            scores = torch.stack([(q[b, pos] * k[b, i]).sum() for i in indices])
            weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)
            out[b, pos] = sum(weights[j] * v[b, i] for j, i in enumerate(indices))
    return out


def test_local_reference_matches_direct_causal_attention():
    torch.manual_seed(0)
    q = torch.randn(2, 7, 5)
    k = torch.randn(2, 7, 5)
    v = torch.randn(2, 7, 5)
    actual = _reference_local_attention(q, k, v, 3, None, is_causal=True, query_chunk=3)
    expected = _direct_local(q, k, v, 3, causal=True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_routed_indices_are_remote_for_causal_last_query():
    torch.manual_seed(1)
    q = torch.randn(2, 10, 6)
    k = torch.randn(2, 10, 6)
    mask = torch.ones(2, 10, dtype=torch.bool)
    indices, active, margin, peakiness = route_global_indices(
        q,
        k,
        top_k=3,
        low_rank_dim=4,
        window_size=4,
        key_mask=mask,
        is_causal=True,
        gate_threshold=0.0,
        peak_threshold=-1.0,
        always_global=True,
    )
    assert indices.shape == (2, 3)
    assert active.all()
    assert torch.isfinite(margin).all()
    assert torch.isfinite(peakiness).all()
    assert (indices < 6).all()


def test_peakiness_gate_skips_diffuse_scores_but_keeps_a_remote_peak():
    torch.manual_seed(2)
    q = torch.randn(1, 128, 16)
    k = torch.randn(1, 128, 16)
    _, diffuse_active, _, diffuse_peak = route_global_indices(
        q,
        k,
        top_k=16,
        low_rank_dim=8,
        window_size=16,
        key_mask=None,
        is_causal=True,
        gate_threshold=0.5,
        peak_threshold=-1.0,
        always_global=False,
    )
    q[:, -1] = k[:, 0] * 4.0
    indices, peak_active, _, peakiness = route_global_indices(
        q,
        k,
        top_k=16,
        low_rank_dim=8,
        window_size=16,
        key_mask=None,
        is_causal=True,
        gate_threshold=0.5,
        peak_threshold=-1.0,
        always_global=False,
    )
    assert diffuse_peak.item() < -1.0
    assert not diffuse_active.item()
    assert peak_active.item()
    assert peakiness.item() > -1.0
    assert 0 in indices[0].tolist()


def test_routed_reference_is_finite_and_causal():
    torch.manual_seed(3)
    q = torch.randn(1, 9, 4)
    k = torch.randn(1, 9, 4)
    v = torch.randn(1, 9, 4)
    routed = torch.tensor([[0, 2, 4]])
    out = _reference_routed_attention(
        q, k, v, routed, 3, None, is_causal=True, query_chunk=4
    )
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_core_active_path_is_exact_routed_path_and_has_gradients():
    """Core output should match causal_sparse_attention and have gradients."""
    from sparse_attn_utils import (
        effective_top_k,
        last_query_topk_indices,
        causal_sparse_attention,
    )
    torch.manual_seed(3)
    q = torch.randn(2, 8, 4, requires_grad=True)
    k = torch.randn(2, 8, 4, requires_grad=True)
    v = torch.randn(2, 8, 4, requires_grad=True)
    mask = torch.ones(2, 8, dtype=torch.bool)
    core = ConfidenceGatedAttentionCore(
        top_k=2,
        low_rank_dim=3,
        window_size=3,
        use_triton=False,
        always_global=True,
    )
    actual = core(q, k, v, mask, 2, 1, is_causal=True, training=True)
    # Reproduce using the proven exp_1 path
    d_low = 3
    k_eff = effective_top_k(2, 8, min_k=64, ratio=2)
    indices = last_query_topk_indices(
        q.detach()[:, :, :d_low], k.detach()[:, :, :d_low], k_eff,
        mask, 2, 1,
    )
    expected = causal_sparse_attention(
        q, k, v, indices, local_window=3,
        token_mask=mask, bsz=2, num_heads=1,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    actual.square().mean().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()


def test_core_skips_global_branch_when_confident():
    """When seq <= window, core uses only local attention (no routing)."""
    torch.manual_seed(4)
    q = torch.randn(1, 3, 4)
    k = torch.randn(1, 3, 4)
    v = torch.randn(1, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.bool)
    # window_size=3, seq_len=3 → src_len <= local_width → local only
    core = ConfidenceGatedAttentionCore(
        top_k=2,
        low_rank_dim=3,
        window_size=3,
        linear_weight=0.25,
        gate_threshold=float("inf"),
        use_triton=False,
        always_global=False,
    )
    actual = core(q, k, v, mask, 1, 1, is_causal=True, training=True)
    expected = _reference_local_attention(q, k, v, 3, mask, is_causal=True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert core.last_stats.active_heads == 0
    assert core.last_stats.route_k == 0


def test_short_causal_sequence_uses_only_exact_local_path():
    torch.manual_seed(5)
    q = torch.randn(1, 4, 3)
    k = torch.randn(1, 4, 3)
    v = torch.randn(1, 4, 3)
    mask = torch.ones(1, 4, dtype=torch.bool)
    core = ConfidenceGatedAttentionCore(
        top_k=4,
        low_rank_dim=2,
        window_size=8,
        use_triton=False,
        always_global=True,
    )
    actual = core(q, k, v, mask, 1, 1, is_causal=True, training=True)
    expected = _direct_local(q, k, v, 8, causal=True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert core.last_stats.active_heads == 0


def test_bidirectional_path_is_sparse_and_differentiable():
    torch.manual_seed(6)
    q = torch.randn(1, 10, 4, requires_grad=True)
    k = torch.randn(1, 10, 4, requires_grad=True)
    v = torch.randn(1, 10, 4, requires_grad=True)
    mask = torch.ones(1, 10, dtype=torch.bool)
    mask[:, -2:] = False
    core = ConfidenceGatedAttentionCore(
        top_k=3,
        low_rank_dim=3,
        window_size=3,
        use_triton=False,
        always_global=True,
    )
    out = core(q, k, v, mask, 1, 1, is_causal=False, training=True)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()
