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
          sm0: tl.constexpr, sm1: tl.constexpr, HAS_MASK: tl.constexpr,
          bh, sk0: tl.constexpr, sk1: tl.constexpr, sk2: tl.constexpr,
          sv0: tl.constexpr, sv1: tl.constexpr, sv2: tl.constexpr,
          ds, D: tl.constexpr, N: tl.constexpr, SCALE: tl.constexpr,
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
               sq0: tl.constexpr, sq1: tl.constexpr, sq2: tl.constexpr,
               sk0: tl.constexpr, sk1: tl.constexpr, sk2: tl.constexpr,
               sv0: tl.constexpr, sv1: tl.constexpr, sv2: tl.constexpr,
               si0: tl.constexpr, si1: tl.constexpr, si2: tl.constexpr,
               sm0: tl.constexpr, sm1: tl.constexpr,
               N: tl.constexpr, T: tl.constexpr, OFFSET: tl.constexpr,
               D: tl.constexpr, H: tl.constexpr,
               FRONT: tl.constexpr, WINDOW: tl.constexpr, ROUTE_K: tl.constexpr,
               ROUTE_CHUNK: tl.constexpr, HAS_MASK: tl.constexpr,
               SCALE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               BD: tl.constexpr):
    block = tl.program_id(0)
    bh = tl.program_id(1)
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
            sm0, sm1, HAS_MASK, bh, sk0, sk1, sk2, sv0, sv1, sv2,
            ds, D, N, SCALE, acc, denom, maximum)
    local_start = tl.maximum(0, block * BM + OFFSET - WINDOW + 1)
    for start in range(tl.cdiv(WINDOW + BM - 1, BN)):
        pos = local_start + start * BN + ns
        valid = ((pos[None, :] >= FRONT) & (pos[None, :] <= qs[:, None])
                 & (pos[None, :] >= qs[:, None] - WINDOW + 1))
        acc, denom, maximum = _step(q, K, V, pos, valid, MASK, bh // H,
            sm0, sm1, HAS_MASK, bh, sk0, sk1, sk2, sv0, sv1, sv2,
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
            sm0, sm1, HAS_MASK, bh, sk0, sk1, sk2, sv0, sv1, sv2,
            ds, D, N, SCALE, acc, denom, maximum)
    out = acc / tl.maximum(denom[:, None], 1.0e-20)
    if HAS_MASK:
        query_ok = tl.load(MASK + (bh // H) * sm0 + qs * sm1, qs < N, other=0)
        out = tl.where(query_ok[:, None], out, 0.0)
    tl.store(OUT + (bh * T + local_qs[:, None]) * D + ds[None, :], out,
             (local_qs[:, None] < T) & (ds[None, :] < D))


def bigger_bird_flash(q, k, v, indices, *, front=64, window=256,
                     token_mask=None, num_heads=32, route_chunk=None,
                     scale=1.0, block_m=64, block_n=64):
    """Q/K/V [BH,N,D]; unique route indices [BH,G,K], -1 marks invalid.

Routing groups must align with query tiles. This kernel only enforces
causal *edges*: the routing policy must separately ensure prefix invariance.
"""
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Triton Bigger Bird requires CUDA FP16/BF16")
    if k.shape != v.shape or q.ndim != 3 or q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Q [BH,T,D] and K/V [BH,N,D] must have matching heads and dimensions")
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
        n, t, offset, d, num_heads, min(front, n), window, indices.shape[-1],
        route_chunk, token_mask is not None, scale, block_m, block_n,
        triton.next_power_of_2(d), num_warps=4, num_stages=2)
    return out
