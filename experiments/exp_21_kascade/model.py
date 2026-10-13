"""Exp 21 — Kascade-style anchor-layer route reuse on Bigger Bird v3.

v3 (`modal_rd.attention.RDAttention`) scores every key column at every layer to
pick each head's routed columns. Here only the anchor layers score; every other
layer reuses the routed columns of the nearest anchor at or before it. Everything
else (sink, window, exact tail, dense cached decode) is v3.

Options (from Kascade):
  head_map  head h of reuse layer j takes the anchor head whose routed columns
            overlapped its own the most on the Phase 0 dev prompts, instead of head h.
  pool      the query heads sharing a KV head average their routing probabilities
            and share one set of columns (8 sets per layer instead of 32).
  decode    also make cached decode sparse with the same anchors / head map / pooling
            (`sparse_decode.py`); otherwise decode stays dense as in v3.

Routing cost drops from 32 scoring passes to len(anchors).
"""
from pathlib import Path

import numpy as np
import torch

from modal_rd import attention as rd


def owner(j, anchors):
    return max(a for a in anchors if a <= j)


def head_map(inter, anchors):
    """[L, H] anchor-head index per (layer, head); identity on anchor layers.

    inter[a, b, h, g] = mean overlap of anchor head (a, h) with head (b, g), from
    `analyze_overlap.py` (matrices.npz). Head g of layer j reuses argmax_h inter[a(j), j, h, g].
    With pooled routes the heads of a KV group have identical rows, so a whole group
    maps to one anchor group.
    """
    L, H = inter.shape[0], inter.shape[2]
    m = np.tile(np.arange(H), (L, 1))
    for j in range(L):
        a = owner(j, anchors)
        if a != j:
            m[j] = inter[a, j].argmax(0)
    return m


def load_head_map(tag, anchors):
    d = Path(__file__).parents[2] / "benchmarks" / "exp_21_kascade" / "phase0" / tag
    return head_map(np.load(d / "matrices.npz")["inter"], anchors)


def pooled_routes(attn, q, k):
    """v3 routes() (last-query, mean pooling) with probabilities averaged over each KV group.

    Returns [H, V] int32 with the same columns for every query head of a KV head.
    """
    n, G = k.shape[-2], attn.groups
    budget = min(rd.vertical_budget(n, attn.vertical, attn.vertical_sqrt), max(0, n - attn.sink - attn.window))
    r = min(attn.n_route, n)
    qpos = torch.arange(n - r, n, device=q.device)
    qa = q[0][:, qpos, :]                                           # [H, r, d]
    ka = k[0].repeat_interleave(G, 0)                               # [H, n, d]
    col = torch.zeros(attn.num_heads, n, device=q.device, dtype=torch.float32)
    keypos = torch.arange(n, device=q.device)
    for h0 in range(0, attn.num_heads, 8):
        s = torch.matmul(qa[h0:h0 + 8], ka[h0:h0 + 8].transpose(-1, -2)).float() * attn.scaling
        s.masked_fill_(keypos[None, None, :] > qpos[None, :, None], -float("inf"))
        col[h0:h0 + 8] = s.softmax(-1).mean(1)
    col = col.view(attn.num_kv_heads, G, n).mean(1)                 # [KV, n]
    col[:, :attn.sink] = -1
    col[:, n - attn.window:] = -1
    idx = col.topk(budget, dim=-1).indices.sort(-1).values.to(torch.int32)
    return idx.repeat_interleave(G, 0)


def install(model, anchors, head_map=None, pool=False, decode=False, **attn):
    """Install v3 on every layer, then make non-anchor layers reuse anchor routes."""
    anchors = sorted(set(anchors))
    if not anchors or anchors[0] != 0:
        raise ValueError("layer 0 must be an anchor")
    if decode:
        from experiments.exp_21_kascade import sparse_decode
        sparse_decode.install(model, anchors, head_map=head_map, pool=pool, **attn)
    else:
        rd.install(model, "vs", **attn)
    shared = {}
    for i, layer in enumerate(model.model.layers):
        attn_mod = layer.self_attn
        if i in anchors:
            if pool:
                def anchor_routes(q, k, _m=attn_mod):
                    shared["idx"] = pooled_routes(_m, q, k)
                    return shared["idx"]
            else:
                def anchor_routes(q, k, _routes=attn_mod.routes):
                    shared["idx"] = _routes(q, k)
                    return shared["idx"]
            attn_mod.routes = anchor_routes
        elif head_map is None:
            attn_mod.routes = lambda q, k: shared["idx"]
        else:
            rows = torch.as_tensor(head_map[i], device=model.device)
            attn_mod.routes = lambda q, k, _rows=rows: shared["idx"][_rows]
    return model
