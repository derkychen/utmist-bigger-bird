"""Fused one-token decode attention over a TurboQuant-style compressed KV cache (modal_rd/turbo_kv.py).

The cache stores, per key/value vector x (head dim d): y = x @ R (R a random orthogonal [d, d]) as 4-bit codes
of y / |y| * sqrt(d) plus |y| in FP16 (byte i: coordinate i in the low nibble, i + d/2 in the high nibble, so the
kernel scores the two halves separately and never interleaves codes); decoding gives y_hat = c[code] * |y| / sqrt(d) and
x_hat = y_hat @ R^T. Attention never rebuilds x_hat:
  scores   q . x_hat = (q @ R) . y_hat          -> rotate the query once per layer
  output   sum_j p_j x_hat_j = (sum_j p_j y_hat_j) @ R^T  -> un-rotate the output once per layer
so the kernel reads only codes and norms (~d/2 + 2 bytes per vector) and decodes them in registers.

Code -> value uses no memory lookup: the codebook is symmetric (c[2^b - 1 - i] = -c[i]), so a code gives a sign
and one of 2^(b-1) magnitudes, chosen with a 3-level select tree (exact, same values as a table lookup). An evenly
spaced codebook (modal_rd.turbo_kv uniform=True) is detected and decoded as (code - offset) * step instead.

Split-K flash-decoding layout as kernels/bigger_bird_decode.py: one program per (KV head, chunk of keys), partial
softmax merged in PyTorch. Two variants for the `groups` query heads of a KV head (4 for Llama-8B): "dot" pads
them to 16 rows for tensor-core tl.dot, "fma" uses plain multiply-adds on the real rows only. `variant="auto"`
times both once per shape and keeps the faster (do that first call outside CUDA-graph capture). `n` may be a
1-element GPU tensor (CUDA-graph safe).
"""
import torch
import triton
import triton.language as tl

_ROUND = 128          # split lengths are multiples of every autotuned BN


@triton.jit
def _value(code, HALF: tl.constexpr, M0, M1, M2, M3, M4, M5, M6, M7, MODE: tl.constexpr):
    """Code -> centroid. MODE 0: symmetric codebook (magnitudes M0 < ... < M[HALF-1]) by select tree;
    1: evenly spaced, (code - M1) * M0; 2 (benchmark ablation, wrong values): plain int->float, no codebook."""
    if MODE == 1:
        return (code.to(tl.float32) - M1) * M0
    if MODE == 2:
        return code.to(tl.float32)
    pos = code >= HALF
    idx = tl.where(pos, code - HALF, HALF - 1 - code)
    b0 = (idx & 1) != 0
    b1 = (idx & 2) != 0
    lo = tl.where(b1, tl.where(b0, M3, M2), tl.where(b0, M1, M0))
    hi = tl.where(b1, tl.where(b0, M7, M6), tl.where(b0, M5, M4))
    v = tl.where((idx & 4) != 0, hi, lo)
    return tl.where(pos, v, -v)


@triton.jit
def _decode_tile(C, N_, base_c, base_n, pos, ok, half, sc1,
                 HALF: tl.constexpr, M0, M1, M2, M3, M4, M5, M6, M7, D: tl.constexpr, MODE: tl.constexpr,
                 NO_CODES: tl.constexpr):
    """-> (first half, second half) of the rotated vectors, each [BN, D/2] fp32. NO_CODES (benchmark ablation)
    skips reading the codes."""
    if NO_CODES:
        b = (pos[:, None] + half[None, :]).to(tl.int32) & 255
    else:
        b = tl.load(C + base_c + pos[:, None] * sc1 + half[None, :], ok[:, None], other=0).to(tl.int32)
    norm = tl.load(N_ + base_n + pos, ok, other=0).to(tl.float32) * (1.0 / tl.sqrt(D * 1.0))
    lo = _value(b & 15, HALF, M0, M1, M2, M3, M4, M5, M6, M7, MODE) * norm[:, None]
    hi = _value(b >> 4, HALF, M0, M1, M2, M3, M4, M5, M6, M7, MODE) * norm[:, None]
    return lo, hi


# The kernel is occupancy-limited at 128 registers per thread (Nsight Compute on H100): include small tiles and
# register caps so the tuner can trade registers for more resident warps.
_CONFIGS = ([triton.Config({"BN": bn}, num_warps=w, num_stages=s, maxnreg=r)
             for bn in (16, 32, 64) for w in (4, 8) for s in (1, 2) for r in (None, 64, 96)]
            + [triton.Config({"BN": 16}, num_warps=2, num_stages=s, maxnreg=r) for s in (1, 2) for r in (None, 64)])


@triton.autotune(configs=_CONFIGS, key=["G", "D", "DOT", "MODE", "NO_CODES"])     # configs that do not fit in shared memory are skipped
@triton.jit
def _partial(Q, KC, KN, VC, VN, PM, PL, PO, N_PTR,
             skc0, skc1, skn0, svc0, svc1, svn0,
             N, SPLIT_LEN, SPLITS, M0, M1, M2, M3, M4, M5, M6, M7,
             HALF: tl.constexpr, N_FROM_PTR: tl.constexpr, DOT: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
             D: tl.constexpr, SCALE: tl.constexpr, MODE: tl.constexpr, NO_CODES: tl.constexpr, BN: tl.constexpr):
    kv = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    if N_FROM_PTR:
        N = tl.load(N_PTR).to(tl.int64)
    rows = tl.arange(0, GP)
    half = tl.arange(0, D // 2)
    qrow = Q + (kv * G + rows)[:, None] * D
    q_lo = tl.load(qrow + half[None, :], (rows < G)[:, None], other=0)               # rotated query, fp32
    q_hi = tl.load(qrow + D // 2 + half[None, :], (rows < G)[:, None], other=0)
    if DOT:                                                                          # cast once, outside the loop
        q_lo = q_lo.to(tl.bfloat16)
        q_hi = q_hi.to(tl.bfloat16)
    j0 = split * SPLIT_LEN
    j1 = tl.minimum(j0 + SPLIT_LEN, N)
    acc_lo = tl.zeros((GP, D // 2), tl.float32)
    acc_hi = tl.zeros((GP, D // 2), tl.float32)
    denom = tl.zeros((GP,), tl.float32)
    maximum = tl.full((GP,), -1.0e20, tl.float32)
    for j in range(j0, j1, BN):
        pos = (j + tl.arange(0, BN)).to(tl.int64)
        ok = pos < j1
        k_lo, k_hi = _decode_tile(KC, KN, kv * skc0, kv * skn0, pos, ok, half, skc1,
                                  HALF, M0, M1, M2, M3, M4, M5, M6, M7, D, MODE, NO_CODES)
        if DOT:
            s = tl.dot(q_lo, tl.trans(k_lo.to(tl.bfloat16)))
            s = tl.dot(q_hi, tl.trans(k_hi.to(tl.bfloat16)), s)
        else:
            s = tl.sum(q_lo[:, None, :] * k_lo[None, :, :], 2) + tl.sum(q_hi[:, None, :] * k_hi[None, :, :], 2)
        s = s * (SCALE * 1.4426950408889634)
        s = tl.where(ok[None, :], s, -float("inf"))
        new_max = tl.maximum(tl.maximum(maximum, tl.max(s, 1)), -1.0e20)
        alpha = tl.exp2(maximum - new_max)
        p = tl.exp2(s - new_max[:, None])
        v_lo, v_hi = _decode_tile(VC, VN, kv * svc0, kv * svn0, pos, ok, half, svc1,
                                  HALF, M0, M1, M2, M3, M4, M5, M6, M7, D, MODE, NO_CODES)
        if DOT:
            pb = p.to(tl.bfloat16)
            acc_lo = tl.dot(pb, v_lo.to(tl.bfloat16), acc_lo * alpha[:, None])
            acc_hi = tl.dot(pb, v_hi.to(tl.bfloat16), acc_hi * alpha[:, None])
        else:
            acc_lo = acc_lo * alpha[:, None] + tl.sum(p[:, :, None] * v_lo[None, :, :], 1)
            acc_hi = acc_hi * alpha[:, None] + tl.sum(p[:, :, None] * v_hi[None, :, :], 1)
        denom = denom * alpha + tl.sum(p, 1)
        maximum = new_max
    out_row = (kv * SPLITS + split) * 16 + rows                                # buffers are laid out 16 rows / program
    tl.store(PM + out_row, maximum)
    tl.store(PL + out_row, denom)
    tl.store(PO + out_row[:, None] * D + half[None, :], acc_lo)
    tl.store(PO + out_row[:, None] * D + D // 2 + half[None, :], acc_hi)


_BEST = {}


def _launch(variant, q, kc, kn, vc, vn, mags, half, scale, n, splits, mode=0, no_codes=False):
    h, d = q.shape
    hk, cap, _ = kc.shape
    g = h // hk
    if splits is None:
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        splits = max(1, min(triton.cdiv(cap, 4 * _ROUND), triton.cdiv(4 * sms, hk)))
    split_len = triton.cdiv(triton.cdiv(cap, splits), _ROUND) * _ROUND
    splits = triton.cdiv(cap, split_len)
    dot = variant == "dot"
    gp = max(16, triton.next_power_of_2(g)) if dot else triton.next_power_of_2(g)
    pm = torch.empty(hk * splits * 16, device=q.device, dtype=torch.float32)
    pl = torch.empty_like(pm)
    po = torch.empty(hk * splits * 16, d, device=q.device, dtype=torch.float32)
    _partial[(hk, splits)](q, kc, kn, vc, vn, pm, pl, po, n if n is not None else pm,
                           kc.stride(0), kc.stride(1), kn.stride(0), vc.stride(0), vc.stride(1), vn.stride(0),
                           cap, split_len, splits, *mags, half, n is not None, dot, g, gp, d, scale, mode, no_codes)
    pm, pl = pm.view(hk, splits, 16)[..., :g], pl.view(hk, splits, 16)[..., :g]
    po = po.view(hk, splits, 16, d)[:, :, :g]
    w = torch.exp2(pm - pm.amax(1, keepdim=True))
    return (po * w[..., None]).sum(1) / (pl * w).sum(1)[..., None]            # [hk, g, d], rotated space


def turbo_decode(q, kc, kn, vc, vn, codebook, rot, *, scale, n=None, splits=None, variant="auto", ablate=None):
    """q [H, d] (one token, unrotated); kc / vc [Hkv, cap, d/2] uint8 packed codes; kn / vn [Hkv, cap] FP16 norms;
    codebook: the 2^b symmetric centroids (b <= 4) as a list of floats (a tensor works outside CUDA graphs);
    rot [d, d] FP32. Returns [H, d] in q's dtype."""
    h, d = q.shape
    hk = kc.shape[0]
    if torch.is_tensor(codebook):
        codebook = codebook.tolist()
    if kc.shape[-1] * 2 != d or h % hk or len(codebook) > 16:
        raise ValueError("shape mismatch")
    half = len(codebook) // 2
    steps = [b - a for a, b in zip(codebook, codebook[1:])]
    if max(steps) - min(steps) < 1e-5 * max(steps):                            # evenly spaced: (code - offset) * step
        mode, mags = 1, [steps[0], (len(codebook) - 1) / 2] + [0.0] * 6
    else:
        mode, mags = 0, list(codebook[half:]) + [0.0] * (8 - half)             # positive magnitudes, ascending
    no_codes = ablate == "no-codes"
    if ablate in ("no-decode", "no-codes"):                                    # benchmark only: wrong values
        mode = 2
    qr = (q.float() @ rot).contiguous()                                        # rotate the query
    if variant == "auto":
        key = (h // hk, d, q.device.index)
        if key not in _BEST:
            from triton.testing import do_bench
            times = {v: do_bench(lambda: _launch(v, qr, kc, kn, vc, vn, mags, half, scale, n, splits, mode, no_codes))
                     for v in ("dot", "fma")}
            _BEST[key] = min(times, key=times.get)
        variant = _BEST[key]
    out = _launch(variant, qr, kc, kn, vc, vn, mags, half, scale, n, splits, mode, no_codes)
    return (out.reshape(h, d) @ rot.T).to(q.dtype)                            # un-rotate once
