"""Triton kernels for modal_rd.kv_int8: per-row symmetric INT8 quantise / dequantise.

Bit-identical to the PyTorch reference in modal_rd/kv_int8.py (exact fp32 division, round-half-even),
but one launch each, which matters at decode: every layer quantises one new token and dequantises its
whole cache on every step.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _quantize_kernel(x_ptr, q_ptr, s_ptr, rows, stride_r, D: tl.constexpr, BR: tl.constexpr,
                     MIN_SCALE: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.arange(0, D)
    m = r[:, None] < rows
    x = tl.load(x_ptr + r[:, None] * stride_r + c[None, :], mask=m, other=0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x), 1) / 127, MIN_SCALE)
    q = tl.minimum(tl.maximum(libdevice.rint(libdevice.div_rn(x, s[:, None])), -127), 127)
    tl.store(q_ptr + r[:, None] * D + c[None, :], q.to(tl.int8), mask=m)
    tl.store(s_ptr + r, s.to(tl.float16), mask=r < rows)


@triton.jit
def _dequantize_kernel(q_ptr, s_ptr, o_ptr, rows, D: tl.constexpr, BR: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.arange(0, D)
    m = r[:, None] < rows
    q = tl.load(q_ptr + r[:, None] * D + c[None, :], mask=m, other=0).to(tl.float32)
    s = tl.load(s_ptr + r, mask=r < rows, other=0).to(tl.float32)
    tl.store(o_ptr + r[:, None] * D + c[None, :], (q * s[:, None]).to(o_ptr.dtype.element_ty), mask=m)


def quantize(x, min_scale, out=None):
    """[..., d] -> int8 [..., d], fp16 scales [..., 1]; one row per leading index. `out` = (q, s): contiguous
    tensors to write into (e.g. the next rows of a cache buffer) instead of allocating."""
    d = x.shape[-1]
    x2 = x.reshape(-1, d)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    rows = x2.shape[0]
    if out is None:
        out = (torch.empty(x.shape, device=x.device, dtype=torch.int8),
               torch.empty(*x.shape[:-1], 1, device=x.device, dtype=torch.float16))
    q, s = out
    assert q.is_contiguous() and s.is_contiguous() and q.numel() == x2.numel() and s.numel() == rows
    _quantize_kernel[(triton.cdiv(rows, 32),)](x2, q, s, rows, x2.stride(0), D=d, BR=32, MIN_SCALE=min_scale)
    return q, s


def dequantize(q, s, dtype):
    q, s = q.contiguous(), s.contiguous()
    out = torch.empty(q.shape, device=q.device, dtype=dtype)
    rows = q.numel() // q.shape[-1]
    _dequantize_kernel[(triton.cdiv(rows, 64),)](q, s, out, rows, D=q.shape[-1], BR=64)
    return out
