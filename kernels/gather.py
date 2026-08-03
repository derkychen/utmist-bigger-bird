"""Gather (selected-branch) sparse attention."""

from __future__ import annotations

import torch

from .common import MODE_GATHER, triton_available
from .online_softmax import _launch


@torch.inference_mode()
def sparse_gather_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    token_idx: torch.Tensor,
    key_mask: torch.Tensor | None = None,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Fused attention over gathered keys. q/k/v [BH,T,D], token_idx int32 [BH,T,M]."""
    if not triton_available():
        raise RuntimeError("Triton CUDA kernels are not available")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    token_idx = token_idx.contiguous().to(torch.int32)
    if key_mask is not None:
        key_mask = key_mask.contiguous()
    return _launch(
        MODE_GATHER,
        q,
        k,
        v,
        token_idx.size(-1),
        token_idx,
        key_mask,
        scale=scale,
    )
