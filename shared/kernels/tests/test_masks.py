"""Tests for shared mask builders."""

from __future__ import annotations

import torch

from shared.kernels.masks import build_gather_key_mask, build_key_mask


def test_build_key_mask_from_2d():
    bsz, num_heads, src_len = 2, 4, 16
    am = torch.ones(bsz, src_len, dtype=torch.bool)
    am[0, -2:] = False
    mask = build_key_mask(am, bsz, num_heads, src_len, device=torch.device("cpu"))
    assert mask is not None
    assert mask.shape == (bsz * num_heads, src_len)
    assert mask[0, -1].item() == False
    assert mask[4, -1].item() == True  # batch 1 unmasked


def test_build_key_mask_none():
    assert build_key_mask(None, 2, 4, 8, device=torch.device("cpu")) is None


def test_build_gather_key_mask():
    bsz, num_heads, tgt_len, m = 2, 2, 4, 3
    src_len = 10
    am = torch.ones(bsz, src_len, dtype=torch.bool)
    am[1, 5] = False
    token_idx = torch.randint(0, src_len, (bsz * num_heads, tgt_len, m))
    mask = build_gather_key_mask(am, bsz, num_heads, tgt_len, token_idx)
    assert mask is not None
    assert mask.shape == (bsz * num_heads, tgt_len, m)
