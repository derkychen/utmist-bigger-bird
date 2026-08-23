"""Shared Triton attention kernels for sparse attention and training."""

from .band import band_attention
from .causal_linear import causal_linear_attention
from .common import (
    MODE_BAND,
    MODE_COMPRESSED,
    MODE_GATHER,
    MODE_ROUTED_WINDOW,
    MODE_WINDOW,
    triton_available,
)
from .dispatch import should_use_train_kernel, should_use_triton, train_kernels_enabled
from .gather import sparse_gather_attention
from .gather_autograd import gather_attention_autograd
from .linear import elu_linear_attention
from .masks import build_gather_key_mask, build_key_mask
from .routed_window import routed_window_attention
from .routed_window_autograd import routed_window_attention_autograd
from .routed_window_fast import routed_window_attention_fast
from .window import sliding_window_attention

__all__ = [
    "MODE_BAND",
    "MODE_COMPRESSED",
    "MODE_GATHER",
    "MODE_ROUTED_WINDOW",
    "MODE_WINDOW",
    "triton_available",
    "should_use_triton",
    "should_use_train_kernel",
    "train_kernels_enabled",
    "sliding_window_attention",
    "routed_window_attention",
    "routed_window_attention_autograd",
    "routed_window_attention_fast",
    "causal_linear_attention",
    "sparse_gather_attention",
    "gather_attention_autograd",
    "band_attention",
    "elu_linear_attention",
    "build_key_mask",
    "build_gather_key_mask",
]
