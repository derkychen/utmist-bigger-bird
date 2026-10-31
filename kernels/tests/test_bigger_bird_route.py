"""Parity tests for fused anchor routing against the PyTorch pooled-softmax top-k."""

from __future__ import annotations

import pytest
import torch

from kernels import triton_available

pytestmark = pytest.mark.skipif(not triton_available(), reason="Triton CUDA kernels not available")


def _reference(q, k, n, front, window, scale):
    hk = k.shape[0]
    g = q.shape[0] // hk
    p = (q.float().view(hk, g, -1) @ k[:, :n].float().transpose(1, 2) * scale).softmax(-1).mean(1)
    p[:, :front] = -1
    p[:, n - window:] = -1
    return p


@pytest.mark.parametrize("cap,n,front,window,routed,dtype,device_n", [
    (5000, 5000, 64, 1024, 256, torch.bfloat16, False),
    (40000, 40000, 64, 4096, 1024, torch.bfloat16, False),
    (40000, 33333, 16, 1000, 512, torch.float16, True),        # length on the GPU, garbage past it
    (131072, 131072, 64, 1024, 1024, torch.bfloat16, False),
])
def test_matches_reference(cap, n, front, window, routed, dtype, device_n):
    from kernels.bigger_bird_route import bigger_bird_route, pooled_probs

    torch.manual_seed(0)
    h, hk, d = 32, 8, 128
    k = torch.randn(1, hk, cap, d, device="cuda", dtype=dtype)[0] * 2
    k[:, n:] = 50
    q = torch.randn(h, d, device="cuda", dtype=dtype)
    scale = d ** -0.5
    kw = dict(front=front, window=window, scale=scale, n=torch.tensor([n], device="cuda") if device_n else None)
    k_in = k if device_n else k[:, :n]
    p = pooled_probs(q, k_in, **kw)
    ref = _reference(q, k, n, front, window, scale)
    torch.testing.assert_close(p[:, :n], ref, atol=1e-6, rtol=2e-3)
    if device_n:
        assert (p[:, n:] == -1).all()
    idx = bigger_bird_route(q, k_in, routed, **kw)
    want = ref.topk(routed, dim=-1).indices
    for a, b in zip(idx.tolist(), want.tolist()):
        assert len(set(a) & set(b)) >= 0.99 * routed          # near-ties may swap at the boundary
