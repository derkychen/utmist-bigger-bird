"""Sliding-window causal attention."""

from __future__ import annotations

import torch

from .common import MODE_WINDOW, triton_available
from .online_softmax import _launch


@torch.inference_mode()
def sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    key_mask: torch.Tensor | None = None,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Causal sliding-window attention. q/k/v [BH,T,D]; key_mask optional [BH,S]."""
    if not triton_available():
        raise RuntimeError("Triton CUDA kernels are not available")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    w = min(window_size, k.size(1))
    if key_mask is not None:
        key_mask = key_mask.contiguous()
    return _launch(
        MODE_WINDOW,
        q,
        k,
        v,
        w,
        None,
        key_mask,
        scale=scale,
        window_size=w,
    )
