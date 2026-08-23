"""Parity tests for the fused local plus routed causal kernel."""

from __future__ import annotations

import pytest
import torch

from experiments.exp_18_confidence_gated.attention_core import _reference_routed_attention
from kernels import routed_window_attention, triton_available


@pytest.mark.skipif(not triton_available(), reason="Triton CUDA kernels not available")
def test_routed_window_attention_parity():
    torch.manual_seed(0)
    bh, seq_len, head_dim = 3, 24, 16
    window = 5
    q = torch.randn(bh, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(bh, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(bh, seq_len, head_dim, device="cuda", dtype=torch.float16)
    routed = torch.tensor([[0, 3, 7], [1, 5, 9], [2, 4, 12]], device="cuda")

    actual = routed_window_attention(q, k, v, routed, window, scale=1.0)
    expected = _reference_routed_attention(
        q.float(),
        k.float(),
        v.float(),
        routed,
        window,
        None,
        is_causal=True,
    ).to(torch.float16)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not triton_available(), reason="Triton CUDA kernels not available")
def test_routed_window_attention_mask_parity():
    torch.manual_seed(1)
    bh, seq_len, head_dim = 2, 20, 16
    window = 4
    q = torch.randn(bh, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(bh, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(bh, seq_len, head_dim, device="cuda", dtype=torch.float16)
    routed = torch.tensor([[0, 4, 8], [1, 5, 9]], device="cuda")
    key_mask = torch.ones(bh, seq_len, device="cuda", dtype=torch.bool)
    key_mask[:, -3:] = False

    actual = routed_window_attention(q, k, v, routed, window, key_mask, scale=1.0)
    expected = _reference_routed_attention(
        q.float(),
        k.float(),
        v.float(),
        routed,
        window,
        key_mask,
        is_causal=True,
    ).to(torch.float16)
    valid_rows = key_mask.unsqueeze(-1)
    torch.testing.assert_close(
        torch.where(valid_rows, actual, torch.zeros_like(actual)),
        torch.where(valid_rows, expected, torch.zeros_like(expected)),
        rtol=2e-2,
        atol=2e-2,
    )
