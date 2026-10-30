"""Fused decode routing for anchor layers (exp 21 / Kascade pooled top-k), implemented in Triton.

For one new query token, picks per KV head the `k_route` cached keys with the highest pooled attention
probability: softmax over all n cached keys separately for each query head, averaged over the query heads
that share the KV head, with the first `front` keys and the last `window` keys excluded (they are always
attended). Same selection rule as `sparse_decode.SparseDecodeAttention.sparse_decode_pooled`.

  pass 1  split-K over the keys, one program per (KV head, chunk): scores for all query heads of the
          group in one tl.dot (each key read once), written as fp32 log2-scaled scores, plus per-chunk
          softmax max / sum per head
  pass 2  merges the chunk statistics, converts scores to probabilities, averages over the group and
          applies the front / window / length masks
  top-k   torch.topk over [Hkv, capacity]

`n` may be a 1-element GPU int tensor holding the current cache length (CUDA-graph safe), as in
`bigger_bird_decode`.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _scores(Q, K, S, PM, PL, N_PTR,
            sq0, sq1, sk0, sk1, sk2,
            N, CAP, SPLIT_LEN, SPLITS,
            N_FROM_PTR: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
            D: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr):
    kv = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    if N_FROM_PTR:
        N = tl.load(N_PTR).to(tl.int64)
    rows = tl.arange(0, GP)
    row_ok = rows < G
    ds = tl.arange(0, D).to(tl.int64)
    q = tl.load(Q + (kv * G + rows)[:, None] * sq0 + ds[None, :] * sq1, row_ok[:, None], other=0)
    j0 = split * SPLIT_LEN
    j1 = tl.minimum(j0 + SPLIT_LEN, N)
    maximum = tl.full((GP,), -1.0e20, tl.float32)
    denom = tl.zeros((GP,), tl.float32)
    for j in range(j0, j1, BN):
        pos = (j + tl.arange(0, BN)).to(tl.int64)
        ok = pos < j1
        keys = tl.load(K + kv * sk0 + pos[None, :] * sk1 + ds[:, None] * sk2, ok[None, :], other=0)
        s = tl.dot(q, keys) * (SCALE * 1.4426950408889634)
        s = tl.where(ok[None, :], s, -float("inf"))
        tl.store(S + ((kv * G + rows)[:, None] * CAP + pos[None, :]), s, row_ok[:, None] & ok[None, :])
        new_max = tl.maximum(tl.maximum(maximum, tl.max(s, 1)), -1.0e20)
        denom = denom * tl.exp2(maximum - new_max) + tl.sum(tl.exp2(s - new_max[:, None]), 1)
        maximum = new_max
    out = (kv * SPLITS + split) * GP + rows
    tl.store(PM + out, maximum)
    tl.store(PL + out, denom)


@triton.jit
def _pool(S, PM, PL, P, N_PTR,
          N, CAP, SPLITS, FRONT, WINDOW,
          N_FROM_PTR: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
          SP: tl.constexpr, BN: tl.constexpr):
    kv = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    if N_FROM_PTR:
        N = tl.load(N_PTR).to(tl.int64)
    rows = tl.arange(0, GP)
    sp = tl.arange(0, SP)
    # merge the per-chunk softmax statistics of each query head (redundantly per program; SPLITS is small)
    at = (kv * SPLITS + sp[:, None]) * GP + rows[None, :]
    st_ok = (sp[:, None] < SPLITS) & (rows[None, :] < G)
    pm = tl.load(PM + at, st_ok, other=-1.0e20)
    pl = tl.load(PL + at, st_ok, other=0)
    m = tl.max(pm, 0)                                                       # [GP]
    inv = 1 / tl.maximum(tl.sum(pl * tl.exp2(pm - m[None, :]), 0), 1.0e-30)
    pos = (block * BN + tl.arange(0, BN)).to(tl.int64)
    valid = (pos < N)[None, :] & (rows < G)[:, None]
    s = tl.load(S + (kv * G + rows)[:, None] * CAP + pos[None, :], valid, other=-float("inf"))
    p = tl.sum(tl.where(valid, tl.exp2(s - m[:, None]) * inv[:, None], 0), 0) / G
    keep = (pos >= FRONT) & (pos < N - WINDOW)
    tl.store(P + kv * CAP + pos, tl.where(keep, p, -1.0), pos < CAP)


def pooled_probs(q, k, *, front=64, window=4096, scale=1.0, n=None, splits=None, block_n=64, num_warps=4):
    """q [H, D], k [Hkv, CAP, D] -> [Hkv, CAP] fp32 pooled probabilities; -1 outside [front, n - window)."""
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("bigger_bird_route requires CUDA FP16/BF16")
    h, d = q.shape
    hk, cap, _ = k.shape
    if h % hk or k.shape[-1] != d:
        raise ValueError("q [H,D] and k [Hkv,CAP,D] must match")
    g = h // hk
    if splits is None:
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        splits = max(1, min(triton.cdiv(cap, 4 * block_n), triton.cdiv(4 * sms, hk), 256))
    split_len = triton.cdiv(triton.cdiv(cap, splits), block_n) * block_n
    splits = triton.cdiv(cap, split_len)
    gp = max(16, triton.next_power_of_2(g))
    s = torch.empty(h, cap, device=q.device, dtype=torch.float32)
    pm = torch.empty(hk * splits * gp, device=q.device, dtype=torch.float32)
    pl = torch.empty_like(pm)
    p = torch.empty(hk, cap, device=q.device, dtype=torch.float32)
    n_ptr = n if n is not None else pm
    _scores[(hk, splits)](q, k, s, pm, pl, n_ptr, *q.stride(), *k.stride(),
                          cap, cap, split_len, splits, n is not None, g, gp, d, block_n, scale,
                          num_warps=num_warps)
    pool_bn = 256
    _pool[(hk, triton.cdiv(cap, pool_bn))](s, pm, pl, p, n_ptr, cap, cap, splits, front, window,
                                           n is not None, g, gp, triton.next_power_of_2(splits), pool_bn,
                                           num_warps=num_warps)
    return p


def bigger_bird_route(q, k, k_route, *, front=64, window=4096, scale=1.0, n=None, **kw):
    """Indices [Hkv, k_route] (unsorted) of the highest pooled-probability keys in [front, n - window)."""
    return pooled_probs(q, k, front=front, window=window, scale=scale, n=n, **kw).topk(
        k_route, dim=-1, sorted=False).indices
