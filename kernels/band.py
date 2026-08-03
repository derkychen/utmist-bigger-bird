"""Symmetric band (local window) softmax attention."""

from __future__ import annotations

import torch

from .common import MODE_BAND, triton_available
from .online_softmax import _launch


@torch.inference_mode()
def band_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    radius: int,
    key_mask: torch.Tensor | None = None,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Fused symmetric band attention: query t attends keys in [t-radius, t+radius].

    q/k/v are [BH, T, D]; key_mask is an optional per-key validity mask [BH, S].
    Unlike :func:`sliding_window_attention`, the band is non-causal (symmetric).
    """
    if not triton_available():
        raise RuntimeError("Triton CUDA kernels are not available")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    src_len = k.size(1)
    radius = min(radius, src_len - 1) if src_len > 0 else 0
    n_keys = 2 * radius + 1
    if key_mask is not None:
        key_mask = key_mask.contiguous()
    return _launch(
        MODE_BAND,
        q,
        k,
        v,
        n_keys,
        None,
        key_mask,
        scale=scale,
        radius=radius,
    )
