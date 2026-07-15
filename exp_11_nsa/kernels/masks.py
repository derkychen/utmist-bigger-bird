"""NSA-specific mask builders for compressed-block attention."""

from __future__ import annotations

import torch


def build_block_ok(
    attention_mask: torch.Tensor | None,
    bsz: int,
    num_heads: int,
    n_cmp: int,
    stride: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Block validity [BH, n_cmp] from block-start token mask."""
    if attention_mask is None:
        return None
    am = attention_mask if attention_mask.dtype == torch.bool else attention_mask > -1e-8
    if am.dim() == 4:
        am = am[:, 0, 0, :]
    elif am.dim() == 2:
        am = am
    block_starts = torch.arange(n_cmp, device=device) * stride
    block_starts = block_starts.clamp(max=am.size(-1) - 1)
    block_ok = am[:, block_starts]
    return block_ok.unsqueeze(1).expand(bsz, num_heads, n_cmp).reshape(bsz * num_heads, n_cmp)
