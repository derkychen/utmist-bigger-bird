"""Dispatch helpers for Triton vs PyTorch fallback."""

from __future__ import annotations

import os

import torch

from .common import triton_available

_TRUTHY = {"1", "true", "yes", "on"}


def should_use_triton(use_triton: bool, q: torch.Tensor, *, training: bool = False) -> bool:
    """True when fused Triton kernels may be used for tensor q (inference-only)."""
    return bool(use_triton and not training and q.is_cuda and triton_available())


def train_kernels_enabled() -> bool:
    """Opt-in flag for the (experimental) autograd-capable training kernels.

    Off by default; enable with env var BIGGER_BIRD_TRAIN_KERNELS=1.
    """
    return os.environ.get("BIGGER_BIRD_TRAIN_KERNELS", "0").lower() in _TRUTHY


def should_use_train_kernel(use_triton: bool, q: torch.Tensor) -> bool:
    """True when the autograd training kernels may run for tensor q."""
    return bool(
        use_triton and train_kernels_enabled() and q.is_cuda and triton_available()
    )
