"""Inference-only tiled sparse FlashAttention, implemented in Triton.

One softmax over the unique union of front anchors, a causal local window,
and routed keys. Keys are loaded directly; no [BH, N, budget, D] gather.
Q may be pre-scaled; callers must provide the corresponding scale explicitly.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _step(q, kp, vp, positions, valid, mask_ptr, batch,
          sm0, sm1, HAS_MASK: tl.constexpr,
          bh, sk0, sk1, sk2,
          sv0, sv1, sv2,
          ds, D: tl.constexpr, N, SCALE: tl.constexpr,
          acc, denom, maximum):
    in_bounds = (positions >= 0) & (positions < N)
    if HAS_MASK:
        key_ok = tl.load(mask_ptr + batch * sm0 + positions * sm1,
                         in_bounds, other=0)
        valid = valid & key_ok[None, :]
    keys = tl.load(kp + bh * sk0 + positions[None, :] * sk1 + ds[:, None] * sk2,
                   in_bounds[None, :] & (ds[:, None] < D), other=0)
    values = tl.load(vp + bh * sv0 + positions[:, None] * sv1 + ds[None, :] * sv2,
                     in_bounds[:, None] & (ds[None, :] < D), other=0)
    scores = tl.dot(q, keys) * (SCALE * 1.4426950408889634)
    scores = tl.where(valid & in_bounds[None, :], scores, -float("inf"))
    new_max = tl.maximum(maximum, tl.max(scores, 1))
    # Finite neutral max also handles padded/all-masked rows without NaNs.
    new_max = tl.maximum(new_max, -1.0e20)
    alpha = tl.exp2(maximum - new_max)
    probabilities = tl.exp2(scores - new_max[:, None])
    acc = acc * alpha[:, None] + tl.dot(probabilities.to(q.dtype), values)
    denom = denom * alpha + tl.sum(probabilities, 1)
    return acc, denom, new_max


@triton.jit
def _attention(Q, K, V, IDX, MASK, OUT,
               # Sequence lengths, strides and WINDOW are runtime values (Triton only specialises them on ==1 and %16),
               # so prompts of new lengths (and the exact tail, which passes window=n) reuse compiled kernels.
               sq0, sq1, sq2, sk0, sk1, sk2, sv0, sv1, sv2, si0, si1, si2, sm0, sm1,
               N, T, OFFSET,
               D: tl.constexpr, H: tl.constexpr, GROUPS: tl.constexpr,
               FRONT: tl.constexpr, WINDOW, ROUTE_K: tl.constexpr,
               ROUTE_CHUNK, HAS_MASK: tl.constexpr,
               SCALE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               BD: tl.constexpr):
    block = tl.program_id(0)
    bh = tl.program_id(1)
    # K/V are stored per KV head: query head h reads KV head h // GROUPS (GROUPS=1: MHA).
    kvbh = (bh // H) * (H // GROUPS) + (bh % H) // GROUPS
    local_qs = block * BM + tl.arange(0, BM)
    qs = local_qs + OFFSET
    ds = tl.arange(0, BD)
    ns = tl.arange(0, BN)
    q = tl.load(Q + bh * sq0 + local_qs[:, None] * sq1 + ds[None, :] * sq2,
                (local_qs[:, None] < T) & (ds[None, :] < D), other=0)
    acc = tl.full((BM, BD), 0, tl.float32)
    denom = tl.full((BM,), 0, tl.float32)
    maximum = tl.full((BM,), -1.0e20, tl.float32)
    for start in range(tl.cdiv(FRONT, BN)):
        pos = start * BN + ns
        valid = (pos[None, :] < FRONT) & (pos[None, :] <= qs[:, None])
        acc, denom, maximum = _step(q, K, V, pos, valid, MASK, bh // H,
            sm0, sm1, HAS_MASK, kvbh, sk0, sk1, sk2, sv0, sv1, sv2,
            ds, D, N, SCALE, acc, denom, maximum)
    local_start = tl.maximum(0, block * BM + OFFSET - WINDOW + 1)
    for start in range(tl.cdiv(WINDOW + BM - 1, BN)):
        pos = local_start + start * BN + ns
        valid = ((pos[None, :] >= FRONT) & (pos[None, :] <= qs[:, None])
                 & (pos[None, :] >= qs[:, None] - WINDOW + 1))
        acc, denom, maximum = _step(q, K, V, pos, valid, MASK, bh // H,
            sm0, sm1, HAS_MASK, kvbh, sk0, sk1, sk2, sv0, sv1, sv2,
            ds, D, N, SCALE, acc, denom, maximum)
    route_group = (block * BM + OFFSET) // ROUTE_CHUNK
    for start in range(tl.cdiv(ROUTE_K, BN)):
        slots = start * BN + ns
        pos = tl.load(IDX + bh * si0 + route_group * si1 + slots * si2,
                      slots < ROUTE_K, other=-1)
        valid = ((slots[None, :] < ROUTE_K) & (pos[None, :] >= FRONT)
                 & (pos[None, :] <= qs[:, None])
                 & (pos[None, :] < qs[:, None] - WINDOW + 1))
        acc, denom, maximum = _step(q, K, V, pos, valid, MASK, bh // H,
            sm0, sm1, HAS_MASK, kvbh, sk0, sk1, sk2, sv0, sv1, sv2,
            ds, D, N, SCALE, acc, denom, maximum)
    out = acc / tl.maximum(denom[:, None], 1.0e-20)
    if HAS_MASK:
        query_ok = tl.load(MASK + (bh // H) * sm0 + qs * sm1, qs < N, other=0)
        out = tl.where(query_ok[:, None], out, 0.0)
    tl.store(OUT + (bh * T + local_qs[:, None]) * D + ds[None, :], out,
             (local_qs[:, None] < T) & (ds[None, :] < D))


def bigger_bird_flash(q, k, v, indices, *, front=64, window=256,
                     token_mask=None, num_heads=32, route_chunk=None, kv_groups=1,
                     scale=1.0, block_m=64, block_n=64, num_warps=4, num_stages=2):
    """Q [BH,T,D], K/V [BH/kv_groups,N,D]; unique route indices [BH,G,K], -1 marks invalid.

`kv_groups` > 1 reads grouped-query K/V in place (query head h uses KV head h // kv_groups),
so callers never materialise a repeat_interleave'd copy of K/V.

Routing groups must align with query tiles. This kernel only enforces
causal *edges*: the routing policy must separately ensure prefix invariance.
"""
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Triton Bigger Bird requires CUDA FP16/BF16")
    if (k.shape != v.shape or q.ndim != 3 or q.shape[0] != k.shape[0] * kv_groups
            or q.shape[-1] != k.shape[-1] or num_heads % kv_groups):
        raise ValueError("Q [BH,T,D] and K/V [BH/kv_groups,N,D] must have matching heads and dimensions")
    if front < 0 or window < 1:
        raise ValueError("front >= 0 and window >= 1 required")
    bh, t, d = q.shape
    n = k.shape[1]
    offset = n - t
    if offset < 0:
        raise ValueError("Key sequence must be at least as long as query sequence")
    if indices.ndim == 2:
        indices = indices[:, None, :]
    route_chunk = route_chunk or max(block_m, triton.next_power_of_2(n))
    if route_chunk % block_m or indices.shape[:2] != (bh, triton.cdiv(n, route_chunk)):
        raise ValueError("Routing groups must cover the sequence and align with query tiles")
    indices = indices.contiguous().to(torch.int32)
    if token_mask is not None:
        if token_mask.shape != (bh // num_heads, n) or token_mask.dtype != torch.bool:
            raise ValueError("token_mask must be boolean [batch,N]")
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _attention[(triton.cdiv(t, block_m), bh)](
        q, k, v, indices, token_mask if token_mask is not None else q, out,
        *q.stride(), *k.stride(), *v.stride(), *indices.stride(),
        *(token_mask.stride() if token_mask is not None else (0, 0)),
        n, t, offset, d, num_heads, kv_groups, min(front, n), window, indices.shape[-1],
        route_chunk, token_mask is not None, scale, block_m, block_n,
        triton.next_power_of_2(d), num_warps=num_warps, num_stages=num_stages)
    return out
