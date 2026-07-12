"""NSA Triton kernels: re-exports shared kernels plus NSA-exclusive ops."""

from shared.kernels import (
    build_gather_key_mask,
    build_key_mask,
    should_use_triton,
    sliding_window_attention,
    sparse_gather_attention,
    triton_available,
)

from .compressed import compressed_causal_attention
from .masks import build_block_ok

__all__ = [
    "triton_available",
    "should_use_triton",
    "sliding_window_attention",
    "compressed_causal_attention",
    "sparse_gather_attention",
    "build_key_mask",
    "build_gather_key_mask",
    "build_block_ok",
]
