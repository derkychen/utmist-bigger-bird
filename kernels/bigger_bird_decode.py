"""Split-K sparse decode attention (one new query token), implemented in Triton.

Attends each query head to front (sink) keys + the last `window` keys + its routed keys, the decode
counterpart of `bigger_bird_flash` (exp 21 sparse decode). Flash-decoding layout: the virtual key list
front | window | routed is cut into `splits` chunks, one program per (route group, chunk) writes a partial
softmax (max, sum, weighted values), and a small PyTorch merge combines the chunks. Keys and values are
read in place from the KV cache (no gathered copy). Positions are derived in-kernel for front / window;
only routed keys come from `indices`.

`n` may be a 1-element int tensor on the GPU holding the current cache length (keys at >= n are ignored),
so a CUDA graph captured once stays valid as the cache grows inside a fixed-capacity buffer. With
front=0, window=capacity and no routed keys this is plain (dense) flash-decoding over the first n keys.

Route groups: with pooled routes (`indices` [Hkv, R]) the `groups` query heads of a KV head share one
key list and each key is loaded once for all of them; with per-head routes (`indices` [H, R]) every
query head is its own group. Routed positions must lie in [front, n - window) and be unique per group.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _partial(Q, K, V, IDX, PM, PL, PO, N_PTR,
             sq0, sq1, sk0, sk1, sk2, sv0, sv1, sv2, si0, si1,
             N, WINDOW, ROUTE_K, SPLIT_LEN, SPLITS,
             N_FROM_PTR: tl.constexpr, FRONT: tl.constexpr, ROWS: tl.constexpr, RP: tl.constexpr, GROUPS: tl.constexpr,
             D: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    if N_FROM_PTR:
        N = tl.load(N_PTR).to(tl.int64)
    kv = (r * ROWS) // GROUPS                       # KV head read by this route group
    rows = tl.arange(0, RP)
    row_ok = rows < ROWS
    ds = tl.arange(0, D).to(tl.int64)
    q = tl.load(Q + (r * ROWS + rows)[:, None] * sq0 + ds[None, :] * sq1, row_ok[:, None], other=0)
    total = FRONT + WINDOW + ROUTE_K
    j0 = split * SPLIT_LEN
    j1 = tl.minimum(j0 + SPLIT_LEN, total)
    acc = tl.zeros((RP, D), tl.float32)
    denom = tl.zeros((RP,), tl.float32)
    maximum = tl.full((RP,), -1.0e20, tl.float32)
    for j in range(j0, j1, BN):
        js = j + tl.arange(0, BN)
        ok = js < j1
        routed = js >= FRONT + WINDOW
        ridx = tl.load(IDX + r * si0 + (js - FRONT - WINDOW).to(tl.int64) * si1, ok & routed, other=-1)
        pos = tl.where(js < FRONT, js, tl.where(routed, ridx, N - WINDOW + (js - FRONT))).to(tl.int64)
        ok = ok & (pos >= 0) & (pos < N)
        keys = tl.load(K + kv * sk0 + pos[None, :] * sk1 + ds[:, None] * sk2, ok[None, :], other=0)
        scores = tl.dot(q, keys) * (SCALE * 1.4426950408889634)
        scores = tl.where(ok[None, :], scores, -float("inf"))
        new_max = tl.maximum(tl.maximum(maximum, tl.max(scores, 1)), -1.0e20)
        alpha = tl.exp2(maximum - new_max)
        p = tl.exp2(scores - new_max[:, None])
        values = tl.load(V + kv * sv0 + pos[:, None] * sv1 + ds[None, :] * sv2, ok[:, None], other=0)
        acc = acc * alpha[:, None] + tl.dot(p.to(values.dtype), values)
        denom = denom * alpha + tl.sum(p, 1)
        maximum = new_max
    out_row = (r * SPLITS + split) * RP + rows
    tl.store(PM + out_row, maximum)
    tl.store(PL + out_row, denom)
    tl.store(PO + out_row[:, None] * D + ds[None, :], acc)


def bigger_bird_decode(q, k, v, indices, *, front=64, window=4096, scale=1.0, n=None, splits=None,
                       block_n=64, num_warps=4, num_stages=2):
    """q [H, D] (one token), k / v [Hkv, N, D] (any strides, e.g. a cache view), indices int [Hkv or H, R].
    `n`: current length as a 1-element GPU int tensor (CUDA-graph safe); default all N keys.

    Returns [H, D] in q's dtype. `splits` defaults to enough chunks for ~4 programs per SM.
    """
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("bigger_bird_decode requires CUDA FP16/BF16")
    h, d = q.shape
    hk, cap, _ = k.shape
    groups = h // hk
    if k.shape != v.shape or h % hk or k.shape[-1] != d:
        raise ValueError("q [H,D] and k/v [Hkv,N,D] must match")
    if n is None and cap < front + window:
        raise ValueError("cache shorter than front + window: use dense attention")
    if indices.shape[0] == hk:
        rows, nr = groups, hk
    elif indices.shape[0] == h:
        rows, nr = 1, h
    else:
        raise ValueError("indices must be [Hkv, R] (pooled) or [H, R] (per head)")
    total = front + window + indices.shape[1]
    if splits is None:
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        splits = max(1, min(triton.cdiv(total, 4 * block_n), triton.cdiv(4 * sms, nr)))
    split_len = triton.cdiv(triton.cdiv(total, splits), block_n) * block_n
    splits = triton.cdiv(total, split_len)
    rp = max(16, triton.next_power_of_2(rows))      # tl.dot needs >= 16 rows
    pm = torch.empty(nr * splits * rp, device=q.device, dtype=torch.float32)
    pl = torch.empty_like(pm)
    po = torch.empty(nr * splits * rp, d, device=q.device, dtype=torch.float32)
    _partial[(nr, splits)](q, k, v, indices, pm, pl, po, n if n is not None else pm,
                           *q.stride(), *k.stride(), *v.stride(), *indices.stride(),
                           cap, window, indices.shape[1], split_len, splits,
                           n is not None, front, rows, rp, groups, d, block_n, scale,
                           num_warps=num_warps, num_stages=num_stages)
    pm, pl, po = pm.view(nr, splits, rp)[..., :rows], pl.view(nr, splits, rp)[..., :rows], \
        po.view(nr, splits, rp, d)[:, :, :rows]
    w = torch.exp2(pm - pm.amax(1, keepdim=True))   # [nr, splits, rows]
    out = (po * w[..., None]).sum(1) / (pl * w).sum(1)[..., None]
    return out.reshape(h, d).to(q.dtype)
