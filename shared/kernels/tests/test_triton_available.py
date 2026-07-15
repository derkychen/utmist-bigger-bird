"""Tests for triton_available and dispatch helpers."""

from __future__ import annotations

import torch

from shared.kernels import should_use_triton, triton_available


def test_triton_available_returns_bool():
    assert isinstance(triton_available(), bool)


def test_should_use_triton_requires_cuda():
    q_cpu = torch.zeros(1, 4, 8)
    assert not should_use_triton(True, q_cpu, training=False)


def test_should_use_triton_skips_training():
    if not torch.cuda.is_available():
        return
    q = torch.zeros(1, 4, 8, device="cuda")
    assert not should_use_triton(True, q, training=True)
