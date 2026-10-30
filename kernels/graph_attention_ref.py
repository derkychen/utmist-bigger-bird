"""Pure-PyTorch reference for graph-routed sparse attention.

Same semantics as bigger_bird_flash_ref but accepts per-query-group routing
indices produced by the graph coordinator. Used for correctness tests and
CPU development.
"""
import torch


def graph_attention_ref(q, k, v, indices, *, front=64, window=256,
                        route_chunk=None, kv_groups=1, num_heads=32, scale=1.0):
    """Reference implementation: front anchors + causal window + graph-routed keys.

    Args:
        q: [BH, T, D] queries.
        k: [BH/kv_groups, N, D] keys.
        v: [BH/kv_groups, N, D] values.
        indices: [BH, G, K] routed token positions per head per group (-1 = invalid).
        front: number of sink/anchor tokens.
        window: local causal window size.
        route_chunk: tokens per routing group (indices columns cover one chunk each).
        kv_groups: GQA group size.
        num_heads: total query heads.
        scale: attention scale factor.

    Returns:
        [BH, T, D] attention output.
    """
    bh, t, d = q.shape
    n = k.shape[1]
    offset = n - t
    if indices.ndim == 2:
        indices = indices[:, None, :]
    route_chunk = route_chunk or n
    front = min(front, n)
    dev = q.device

    qs = torch.arange(t, device=dev) + offset
    kp = torch.arange(n, device=dev)
    causal = kp[None, :] <= qs[:, None]
    base = causal & ((kp[None, :] < front) | (kp[None, :] > qs[:, None] - window))
    beyond_window = kp[None, :] < qs[:, None] - window + 1
    group_of_query = qs // route_chunk

    out = torch.zeros(q.shape, dtype=q.dtype, device=dev)
    for h in range(bh):
        kv_h = (h // num_heads) * (num_heads // kv_groups) + (h % num_heads) // kv_groups
        s = (q[h].float() @ k[kv_h].float().T) * scale

        idx = indices[h].long()
        routed = torch.zeros(idx.shape[0], n + 1, dtype=torch.bool, device=dev)
        routed.scatter_(1, torch.where(idx >= 0, idx, n), True)
        routed = routed[:, :n]
        routed_per_q = routed[group_of_query]

        mask = base | (routed_per_q & causal & (kp[None, :] >= front) & beyond_window)
        s = s.masked_fill(~mask, float("-inf"))
        p = torch.softmax(s, -1).nan_to_num(0.0)
        out[h] = (p @ v[kv_h].float()).to(q.dtype)
    return out
