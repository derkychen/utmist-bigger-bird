"""Sparse-prefill / dense-decode attention arms for Llama R&D on Modal.

`vs` (vertical + sink + local) per layer and head:
  * sink:      first `sink` tokens, exact (attention-sink protection)
  * local:     causal sliding window of `window` tokens, exact
  * vertical:  `vertical` key columns chosen per head by the mean (or max,
               `route_pool`) attention probability of `n_route` prompt queries:
               the last ones (`route_queries="last"`) or evenly spread ("spread")
  * exact_tail: the final `exact_tail` prompt queries attend exactly, O(r * n)
  * vertical_scope: if > 0, only queries in the final ~`vertical_scope` tokens
               (rounded to `scope_chunk` groups) use the routed columns
One joint softmax over the union via the existing Triton kernel
(kernels/bigger_bird_flash.py). Prefill attention work is
O(n * (sink + window + vertical)) plus O(n_route * n) routing: linear in n for
fixed budgets. Decode uses exact attention over the full KV cache (O(n) per
token). Routing uses the final prompt queries, so prompt routing is not
prefix-invariant (same limitation as exp19 `last_query`).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from kernels.bigger_bird_flash import bigger_bird_flash


def vertical_budget(n, vertical, vertical_sqrt):
    """Fixed budget, or c*sqrt(n) (rounded to 64) when vertical_sqrt=c > 0."""
    if vertical_sqrt > 0:
        return int(math.ceil(vertical_sqrt * math.sqrt(n) / 64) * 64)
    return vertical


class RDAttention(nn.Module):
    def __init__(self, base, *, sink=64, window=4096, vertical=2048, vertical_sqrt=0.0,
                 n_route=64, route_pool="mean", route_queries="last", exact_tail=0,
                 vertical_scope=0, scope_chunk=4096, dense=False, block_m=64, block_n=64,
                 num_warps=4, num_stages=2):
        super().__init__()
        cfg = base.config
        self.layer_idx = base.layer_idx
        self.head_dim = base.head_dim
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.groups = self.num_heads // self.num_kv_heads
        self.scaling = base.scaling
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = base.q_proj, base.k_proj, base.v_proj, base.o_proj
        self.sink, self.window, self.vertical, self.vertical_sqrt = sink, window, vertical, vertical_sqrt
        if route_pool not in ("mean", "max"):
            raise ValueError("route_pool must be mean or max")
        self.n_route, self.route_pool, self.dense, self.block_m, self.block_n = n_route, route_pool, dense, block_m, block_n
        if route_queries not in ("last", "spread"):
            raise ValueError("route_queries must be last or spread")
        if scope_chunk % block_m or scope_chunk & (scope_chunk - 1):
            raise ValueError("scope_chunk must be a power of two multiple of block_m")
        self.exact_tail, self.route_queries = exact_tail, route_queries
        self.vertical_scope, self.scope_chunk = vertical_scope, scope_chunk
        self.num_warps, self.num_stages = num_warps, num_stages
        self.last_routes = None

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kwargs):
        b, t, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        n = k.shape[-2]
        if b != 1:
            raise ValueError("batch size 1 only")
        if self.dense or t < n or n <= self.sink + self.window:
            # Exact attention: dense layers, cached decode (t < n), or short prompts.
            out = F.scaled_dot_product_attention(q, k, v, is_causal=t == n and t > 1,
                                                 scale=self.scaling, enable_gqa=True)
        else:
            out = self.sparse_prefill(q, k, v)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out), None

    def routes(self, q, k):
        """Per-head top-V key columns by mean softmax prob of the last n_route queries."""
        n = k.shape[-2]
        budget = min(vertical_budget(n, self.vertical, self.vertical_sqrt), max(0, n - self.sink - self.window))
        r = min(self.n_route, n)
        if self.route_queries == "spread":
            # Queries evenly spaced over the prompt: globally attended columns, not question-specific.
            qpos = torch.linspace(self.sink + self.window, n - 1, r, device=q.device).long()
        else:
            qpos = torch.arange(n - r, n, device=q.device)
        qa = q[0][:, qpos, :]                                           # [H, r, d]
        ka = k[0].repeat_interleave(self.groups, 0)                     # [H, n, d]
        col = torch.zeros(self.num_heads, n, device=q.device, dtype=torch.float32)
        keypos = torch.arange(n, device=q.device)
        for h0 in range(0, self.num_heads, 8):
            s = torch.matmul(qa[h0:h0 + 8], ka[h0:h0 + 8].transpose(-1, -2)).float() * self.scaling
            s.masked_fill_(keypos[None, None, :] > qpos[None, :, None], -float("inf"))
            p = s.softmax(-1)
            col[h0:h0 + 8] = p.amax(1) if self.route_pool == "max" else p.mean(1)
        col[:, :self.sink] = -1
        col[:, n - self.window:] = -1                                   # handled by the window branch
        idx = col.topk(budget, dim=-1).indices.sort(-1).values.to(torch.int32)
        return idx

    def sparse_prefill(self, q, k, v):
        n = k.shape[-2]
        idx = self.routes(q, k)
        self.last_routes = idx
        bh = self.num_heads
        Q = (q[0] * self.scaling).contiguous()                          # [H, n, d], pre-scaled
        K = k[0].repeat_interleave(self.groups, 0).contiguous()
        V = v[0].repeat_interleave(self.groups, 0).contiguous()
        if self.vertical_scope:
            # Only query chunks overlapping the final `vertical_scope` tokens use routed columns;
            # earlier queries see sink + local only.
            chunk = self.scope_chunk
            groups = -(-n // chunk)
            starts = torch.arange(groups, device=q.device) * chunk
            active = (starts + chunk > n - self.vertical_scope)[None, :, None]
            routed = torch.where(active, idx[:, None, :], torch.full_like(idx[:, None, :], -1))
        else:
            chunk, routed = None, idx.view(bh, 1, -1)
        out = bigger_bird_flash(Q, K, V, routed.contiguous(), front=self.sink, window=self.window,
                                token_mask=None, num_heads=self.num_heads, route_chunk=chunk, scale=1.0,
                                block_m=self.block_m, block_n=self.block_n,
                                num_warps=self.num_warps, num_stages=self.num_stages)[None]
        r = min(self.exact_tail, n)
        if r:
            # Final r prompt queries (question / answer cue) attend exactly: O(r * n).
            for s in range(n - r, n, 256):   # 256-query chunks bound the masked-SDPA workspace
                e = min(s + 256, n)
                causal = torch.arange(n, device=q.device)[None, :] <= torch.arange(s, e, device=q.device)[:, None]
                out[:, :, s:e] = F.scaled_dot_product_attention(q[:, :, s:e], k, v, attn_mask=causal,
                                                                 scale=self.scaling, enable_gqa=True)
        return out


def install(model, arm, dense_layers=(), **kw):
    """Patch every decoder layer; `dense_layers` stay exact (ablation only)."""
    for i, layer in enumerate(model.model.layers):
        base = layer.self_attn
        base = getattr(base, "_rd_base", base)
        new = RDAttention(base, dense=(arm == "dense_rd" or i in dense_layers), **kw)
        new._rd_base = base
        layer.self_attn = new
    return model
