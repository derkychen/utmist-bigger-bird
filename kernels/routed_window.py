"""Fused causal local-window plus head-shared routed attention."""

from __future__ import annotations

import torch

from .common import MODE_ROUTED_WINDOW, triton_available
from .online_softmax import _launch


@torch.inference_mode()
def routed_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    routed_indices: torch.Tensor,
    window_size: int,
    key_mask: torch.Tensor | None = None,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Causal attention over a local window and head-shared routed keys.

    ``q``, ``k``, and ``v`` are ``[BH, T, D]``. ``routed_indices`` is
    ``[BH, K]`` and is shared across query positions. The kernel applies a
    causal mask to both the local and routed portions without materializing a
    ``[BH, T, window+K]`` index tensor.
    """
    if not triton_available():
        raise RuntimeError("Triton CUDA kernels are not available")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    routed_indices = routed_indices.contiguous().to(torch.int32).unsqueeze(1)
    if key_mask is not None:
        key_mask = key_mask.contiguous()
    window_size = min(int(window_size), k.size(1))
    return _launch(
        MODE_ROUTED_WINDOW,
        q,
        k,
        v,
        window_size + routed_indices.size(-1),
        routed_indices,
        key_mask,
        scale=scale,
        window_size=window_size,
    )
