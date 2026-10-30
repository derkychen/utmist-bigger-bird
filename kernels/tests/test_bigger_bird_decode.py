"""Parity tests for the split-K sparse decode kernel against fp32 attention over the same keys."""

from __future__ import annotations

import pytest
import torch

from kernels import triton_available

pytestmark = pytest.mark.skipif(not triton_available(), reason="Triton CUDA kernels not available")


def _reference(q, k, v, idx, front, window, scale):
    h, d = q.shape
    hk, n, _ = k.shape
    g = h // hk
    rows = idx.shape[0]
    pos = torch.cat([torch.arange(front, device="cuda").expand(rows, -1),
                     torch.arange(n - window, n, device="cuda").expand(rows, -1), idx.long()], 1)
    kv = torch.arange(rows, device="cuda") // (rows // hk)
    ks, vs = k[kv[:, None], pos].float(), v[kv[:, None], pos].float()      # [rows, S, d]
    qr = q.float().view(rows, h // rows, d)
    att = (qr @ ks.transpose(1, 2) * scale).softmax(-1)
    return (att @ vs).reshape(h, d)


@pytest.mark.parametrize("n,front,window,routed,pooled,splits,dtype", [
    (4096, 64, 1024, 256, True, None, torch.bfloat16),
    (4096, 64, 1024, 256, False, None, torch.bfloat16),
    (20000, 64, 4096, 512, True, 7, torch.float16),
    (20000, 16, 333, 1000, False, 1, torch.bfloat16),      # odd window, single split
    (70000, 64, 16384, 512, True, None, torch.bfloat16),
])
def test_matches_reference(n, front, window, routed, pooled, splits, dtype):
    from kernels.bigger_bird_decode import bigger_bird_decode

    torch.manual_seed(0)
    h, hk, d = 32, 8, 128
    cache = torch.randn(2, 1, hk, n, d, device="cuda", dtype=dtype)
    k, v = cache[0, 0] * 2, cache[1, 0]                                     # strided views, as in a KV cache
    q = torch.randn(h, d, device="cuda", dtype=dtype)
    rows = hk if pooled else h
    idx = torch.stack([front + torch.randperm(n - window - front, device="cuda")[:routed] for _ in range(rows)])
    scale = d ** -0.5
    out = bigger_bird_decode(q, k, v, idx, front=front, window=window, scale=scale, splits=splits)
    ref = _reference(q, k, v, idx, front, window, scale)
    assert out.dtype == dtype and out.shape == (h, d)
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)


def test_device_length_and_dense_mode():
    """n as a GPU tensor inside a larger buffer; front=0 / window=capacity / no routes = dense decode."""
    from kernels.bigger_bird_decode import bigger_bird_decode

    torch.manual_seed(1)
    h, hk, d, cap, n = 32, 8, 128, 9000, 7777
    k = torch.randn(hk, cap, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(hk, cap, d, device="cuda", dtype=torch.bfloat16)
    k[:, n:] = 100                                                         # garbage beyond n must be ignored
    q = torch.randn(h, d, device="cuda", dtype=torch.bfloat16)
    length = torch.tensor([n], device="cuda")
    scale = d ** -0.5
    idx = torch.stack([64 + torch.randperm(n - 1024 - 64, device="cuda")[:300] for _ in range(hk)])
    sparse = bigger_bird_decode(q, k, v, idx, front=64, window=1024, scale=scale, n=length)
    ref = _reference(q, k[:, :n], v[:, :n], idx, 64, 1024, scale)
    torch.testing.assert_close(sparse.float(), ref, atol=1e-2, rtol=1e-2)
    none = torch.empty(hk, 0, dtype=torch.long, device="cuda")
    dense = bigger_bird_decode(q, k, v, none, front=0, window=cap, scale=scale, n=length)
    exact = torch.nn.functional.scaled_dot_product_attention(
        q[None, :, None].float(), k[None, :, :n].float(), v[None, :, :n].float(), scale=scale, enable_gqa=True)[0, :, 0]
    torch.testing.assert_close(dense.float(), exact, atol=1e-2, rtol=1e-2)


def test_rejects_short_cache():
    from kernels.bigger_bird_decode import bigger_bird_decode

    k = torch.zeros(8, 100, 128, device="cuda", dtype=torch.bfloat16)
    q = torch.zeros(32, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        bigger_bird_decode(q, k, k, torch.zeros(8, 4, dtype=torch.long, device="cuda"), front=64, window=64)
