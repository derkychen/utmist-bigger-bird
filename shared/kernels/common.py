"""Shared utilities for Triton attention kernels."""

from __future__ import annotations

import os
import sysconfig

import torch

_TRITON_IMPORTED = False
_TRITON_READY: bool | None = None

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    _TRITON_IMPORTED = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

MODE_GATHER = 0
MODE_WINDOW = 1
MODE_COMPRESSED = 2
MODE_BAND = 3

# Match torch.finfo(torch.float32).min used in model.py masked_fill
MASK_SCORE_FP32 = -3.4028234663852886e38


def ceil_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _has_python_dev_headers() -> bool:
    include = sysconfig.get_path("include")
    return os.path.isfile(os.path.join(include, "Python.h"))


def triton_available() -> bool:
    """True if Triton can compile and run a minimal kernel on the current CUDA device."""
    global _TRITON_READY
    if _TRITON_READY is not None:
        return _TRITON_READY
    if not _TRITON_IMPORTED or not torch.cuda.is_available() or not _has_python_dev_headers():
        _TRITON_READY = False
        return False
    try:
        from .online_softmax import _launch

        q = torch.zeros(1, 1, 8, device="cuda", dtype=torch.float16)
        k = torch.zeros(1, 8, 8, device="cuda", dtype=torch.float16)
        v = torch.zeros(1, 8, 8, device="cuda", dtype=torch.float16)
        _launch(MODE_WINDOW, q, k, v, 4, None, None, scale=1.0, window_size=4)
        _TRITON_READY = True
    except Exception:
        _TRITON_READY = False
    return _TRITON_READY
