"""Pure-PyTorch reference for `kernels.bigger_bird_flash.bigger_bird_flash`.

Same arguments and semantics (front anchors + causal local window + routed columns, one
joint softmax), but materialises masked scores, so it is for correctness tests and
Triton-less dev machines (CPU/MPS), not for long contexts.
"""
import torch


def bigger_bird_flash_ref(q, k, v, indices, *, front=64, window=256, token_mask=None,
                          num_heads=32, route_chunk=None, kv_groups=1, scale=1.0,
                          head_chunk=4, **_unused_tile_args):
    bh, t, d = q.shape
    n = k.shape[1]
    if k.shape != v.shape or bh != k.shape[0] * kv_groups or num_heads % kv_groups:
        raise ValueError("Q [BH,T,D] and K/V [BH/kv_groups,N,D] must have matching heads")
    offset = n - t
    if offset < 0:
        raise ValueError("Key sequence must be at least as long as query sequence")
    if indices.ndim == 2:
        indices = indices[:, None, :]
    route_chunk = route_chunk or n
    front = min(front, n)
    dev = q.device
    qs = torch.arange(t, device=dev) + offset                       # absolute query positions
    kp = torch.arange(n, device=dev)
    causal = kp[None, :] <= qs[:, None]                             # [T, N]
    base = causal & ((kp[None, :] < front) | (kp[None, :] > qs[:, None] - window))
    beyond_window = kp[None, :] < qs[:, None] - window + 1
    group_of_query = qs // route_chunk                              # [T]
    out = torch.zeros(q.shape, dtype=q.dtype, device=dev)
    for h0 in range(0, bh, head_chunk):
        h1 = min(h0 + head_chunk, bh)
        heads = torch.arange(h0, h1, device=dev)
        kv = (heads // num_heads) * (num_heads // kv_groups) + (heads % num_heads) // kv_groups
        s = torch.einsum("htd,hnd->htn", q[h0:h1].float(), k[kv].float()) * scale
        idx = indices[h0:h1].long()                                 # [h, G, K]
        routed = torch.zeros(h1 - h0, idx.shape[1], n + 1, dtype=torch.bool, device=dev)
        routed.scatter_(2, torch.where(idx >= 0, idx, n), True)     # slot n swallows -1
        routed = routed[:, :, :n][:, group_of_query]                # [h, T, N]
        mask = base[None] | (routed & causal[None] & (kp[None, None, :] >= front)
                             & beyond_window[None])
        if token_mask is not None:
            b = heads // num_heads
            mask = mask & token_mask[b][:, None, :]
        s = s.masked_fill(~mask, -float("inf"))
        p = torch.softmax(s, -1).nan_to_num(0.0)                    # fully masked row -> 0
        o = torch.einsum("htn,hnd->htd", p, v[kv].float())
        if token_mask is not None:
            o = o * token_mask[b][:, qs].unsqueeze(-1)
        out[h0:h1] = o.to(q.dtype)
    return out
