"""Smoke tests for block-wise teleport routing."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_15_bigger_bird.config import BiggerBirdConfig
from exp_15_bigger_bird.model import BiggerBirdAttention


def _make_attention():
    config = BiggerBirdConfig(
        hidden_size=64,
        num_attention_heads=4,
        intermediate_size=128,
        block_size=16,
        num_random_blocks=2,
        teleports_per_head=3,
        teleport_bias_frac=0.4,
        use_teleports=True,
        use_random_attn=False,
        use_dynamic_globals=False,
        use_topk_mmr=False,
    )
    return BiggerBirdAttention(config)


def test_select_teleport_blocks_shape():
    attn = _make_attention()
    B, H, block_size, d = 2, 4, 16, 64
    num_blocks = 8
    n_mid = num_blocks - 4

    query = torch.randn(B, H, n_mid, block_size, d)
    blocked_key = torch.randn(B, H, num_blocks, block_size, d)
    to_blocked_mask = torch.ones(B, num_blocks, block_size)

    idx = attn.select_teleport_blocks(
        query=query,
        blocked_key_matrix=blocked_key,
        to_blocked_mask=to_blocked_mask,
        num_teleport_blocks=3,
        to_block_size=block_size,
    )
    assert idx.shape == (B, H, n_mid, 3)
    assert idx.min() >= 0
    assert idx.max() < num_blocks
    # static global blocks excluded from routing
    assert not (idx == 0).any()
    assert not (idx == num_blocks - 1).any()


def test_teleport_gather_matches_random_attn_layout():
    attn = _make_attention()
    B, H, block_size, d = 1, 2, 16, 64
    num_blocks = 10
    n_mid = num_blocks - 4
    t_blocks = 3

    blocked_key = torch.randn(B, H, num_blocks, block_size, d)
    tele_idx = torch.randint(1, num_blocks - 1, (B, H, n_mid, t_blocks))

    gathered = attn.torch_gather_b2(blocked_key, tele_idx)
    assert gathered.shape == (B, H, n_mid * t_blocks, block_size, d)

    viewed = gathered.view(B, H, n_mid, t_blocks * block_size, d)
    assert viewed.shape == (B, H, n_mid, t_blocks * block_size, d)


if __name__ == "__main__":
    test_select_teleport_blocks_shape()
    test_teleport_gather_matches_random_attn_layout()
    print("ok")
