"""Weight-only FP8 matrix-vector product for one-token decode, implemented in Triton.

y[n] = scale[n] * sum_k x[k] * w[n, k] with w stored as FP8 E4M3 (one byte per weight, `w_u8` is its uint8 view)
and x in BF16/FP16. Decode at batch 1 reads every weight once per token, so halving the bytes read roughly
halves the time of the linear layers. Weights are converted to FP32 in registers: natively on GPUs with FP8
support (sm89+), bit by bit elsewhere (for tests).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3_to_f32(b):
    """FP8 E4M3 (fn) bit pattern held in uint8 -> float32 (no NaN handling; quantized weights are finite)."""
    b = b.to(tl.int32)
    e = (b >> 3) & 15
    m = (b & 7).to(tl.float32)
    mag = tl.where(e == 0, m * 0.001953125, (m + 8.0) * tl.exp2((e - 10).to(tl.float32)))   # 2^-9; (8+m) 2^(e-10)
    return tl.where((b & 128) != 0, -mag, mag)


@triton.autotune(configs=[triton.Config({"BN": bn, "BK": bk}, num_warps=w)
                          for bn in (4, 8, 16, 32) for bk in (256, 512, 1024) for w in (2, 4, 8)],
                 key=["N", "K"])
@triton.jit
def _gemv(X, W, S, Y, N, K, swn, NATIVE: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BN + tl.arange(0, BN)
    row_ok = rows < N
    acc = tl.zeros((BN,), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        k_ok = ks < K
        x = tl.load(X + ks, k_ok, other=0).to(tl.float32)
        wb = tl.load(W + rows[:, None].to(tl.int64) * swn + ks[None, :], row_ok[:, None] & k_ok[None, :], other=0)
        if NATIVE:
            w = wb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        else:
            w = _e4m3_to_f32(wb)
        acc += tl.sum(w * x[None, :], 1)
    s = tl.load(S + rows, row_ok, other=0)
    tl.store(Y + rows, (acc * s).to(Y.dtype.element_ty), row_ok)


def fp8_gemv(x, w_u8, scale):
    """x [K] (BF16/FP16), w_u8 [N, K] uint8 (FP8 E4M3 bits, row-major), scale [N] FP32 -> y [N] in x's dtype.

    Block sizes are autotuned per (N, K) on first use (outside CUDA-graph capture: call once eagerly first)."""
    n, k = w_u8.shape
    if x.shape[-1] != k or w_u8.stride(1) != 1:
        raise ValueError("x [K] and row-major w [N, K] required")
    y = torch.empty(n, device=x.device, dtype=x.dtype)
    native = torch.cuda.get_device_capability(x.device) >= (8, 9)
    _gemv[lambda meta: (triton.cdiv(n, meta["BN"]),)](x.reshape(-1), w_u8, scale, y, n, k, w_u8.stride(0), native)
    return y


@triton.jit
def _dequant(W, S, O, NUMEL, K, NATIVE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    ok = i < NUMEL
    wb = tl.load(W + i, ok, other=0)
    if NATIVE:
        w = wb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        w = _e4m3_to_f32(wb)
    s = tl.load(S + i // K, ok, other=0)
    tl.store(O + i, (w * s).to(O.dtype.element_ty), ok)


def fp8_dequant(w_u8, scale, dtype=torch.bfloat16):
    """[N, K] FP8 bits (row-major) and [N] FP32 row scales -> [N, K] `dtype` in one pass."""
    n, k = w_u8.shape
    out = torch.empty(n, k, device=w_u8.device, dtype=dtype)
    native = torch.cuda.get_device_capability(w_u8.device) >= (8, 9)
    block = 4096
    _dequant[(triton.cdiv(n * k, block),)](w_u8, scale, out, n * k, k, native, block)
    return out
