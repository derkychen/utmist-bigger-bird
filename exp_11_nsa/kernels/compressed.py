"""Compressed-block causal attention (NSA-specific)."""

from __future__ import annotations

import torch

from shared.kernels.common import MODE_COMPRESSED, triton_available
from shared.kernels.online_softmax import _launch


@torch.inference_mode()
def compressed_causal_attention(
    q: torch.Tensor,
    k_cmp: torch.Tensor,
    v_cmp: torch.Tensor,
    block_size: int,
    stride: int,
    block_ok: torch.Tensor | None = None,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Attention over compressed keys with per-query causal block mask."""
    if not triton_available():
        raise RuntimeError("Triton CUDA kernels are not available")
    q, k_cmp, v_cmp = q.contiguous(), k_cmp.contiguous(), v_cmp.contiguous()
    n_cmp = k_cmp.size(1)
    if block_ok is not None:
        block_ok = block_ok.contiguous()
    return _launch(
        MODE_COMPRESSED,
        q,
        k_cmp,
        v_cmp,
        n_cmp,
        None,
        block_ok,
        scale=scale,
        block_size=block_size,
        stride_blocks=stride,
    )
