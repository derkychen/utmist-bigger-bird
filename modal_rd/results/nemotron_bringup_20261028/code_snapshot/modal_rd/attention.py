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

`kv_evict=True` (sparse prefill only): once a layer's prefill attention is done, its KV cache is
rewritten to just the keys the sparse pattern can ever use for that KV head: sink + last `window`
+ the head group's routed columns (padded to a rectangular `g * budget` with the next-best columns).
Prefill output is unchanged; cached decode then attends only those keys, so the cache shrinks from
n to ~sink + window + g * budget tokens per layer. Question-conditioned: a different question
on the same context needs a fresh prefill.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, rotate_half

from kernels.bigger_bird_flash_ref import bigger_bird_flash_ref

try:
    from kernels.bigger_bird_flash import bigger_bird_flash as _triton_flash
except ImportError:  # no Triton (CPU/MPS dev box)
    _triton_flash = None


def bigger_bird_flash(q, *args, **kwargs):
    """Triton kernel for CUDA FP16/BF16; otherwise the slow pure-PyTorch reference (same semantics)."""
    if _triton_flash is not None and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        return _triton_flash(q, *args, **kwargs)
    return bigger_bird_flash_ref(q, *args, **kwargs)


ROPE_CHUNK = 8192


def rope_inplace(x, cos, sin, chunk=ROPE_CHUNK):
    """`apply_rotary_pos_emb` on x [b, h, t, d] in place, `chunk` positions at a time.

    RoPE is position-wise, so this is bit-identical to the HF version, but the transients are
    chunk-sized instead of 3-4 full copies of Q (~1.6 GB at 65K tokens). x must be a tensor we own."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    for s in range(0, x.shape[-2], chunk):
        xs = x[..., s:s + chunk, :]
        xs.copy_(xs * cos[..., s:s + chunk, :] + rotate_half(xs) * sin[..., s:s + chunk, :])
    return x


def vertical_budget(n, vertical, vertical_sqrt):
    """Fixed budget, or c*sqrt(n) (rounded to 64) when vertical_sqrt=c > 0."""
    if vertical_sqrt > 0:
        return int(math.ceil(vertical_sqrt * math.sqrt(n) / 64) * 64)
    return vertical


def forward_pool(score, p, back=0):
    """[H, n] -> [H, n]: each key takes the best score among itself, the p - 1 keys before it and the `back`
    keys after it, so the continuation of an important token (the rest of a copied number or span) and the
    text leading up to it (e.g. the key a value is bound to) rank with it."""
    p = max(p, 1)
    return F.max_pool1d(F.pad(score[:, None], (p - 1, back), value=-float("inf")), p + back, stride=1)[:, 0]


class RDAttention(nn.Module):
    def __init__(self, base, *, sink=64, window=4096, vertical=2048, vertical_sqrt=0.0,
                 n_route=64, route_pool="mean", route_queries="last", exact_tail=0,
                 vertical_scope=0, scope_chunk=4096, kv_evict=False, evict_keep=0, evict_pool=0, evict_obs=0,
                 evict_obs_max=False, evict_pool_back=0, evict_window=0, evict_full_layers=0, dense=False, block_m=64, block_n=64,
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
        self.kv_evict = kv_evict
        self.evict_keep, self.evict_pool, self.evict_obs = evict_keep, evict_pool, evict_obs
        self.evict_full_layers, self.evict_window = evict_full_layers, evict_window
        self.evict_obs_max, self.evict_pool_back = evict_obs_max, evict_pool_back
        self.last_routes = None
        self.kept = None            # [Hkv, m] absolute positions left in the cache after eviction
        self._col = None

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kwargs):
        b, t, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if torch.is_grad_enabled() and q.requires_grad:
            q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        else:
            q, k = rope_inplace(q, *position_embeddings), rope_inplace(k, *position_embeddings)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        n = k.shape[-2]
        if b != 1:
            raise ValueError("batch size 1 only")
        if self.dense or t < n or n <= self.sink + self.window:
            # Exact attention: dense layers, cached decode (t < n), or short prompts.
            out = self.exact_attention(q, k, v, t, n)
        else:
            out = self.sparse_prefill(q, k, v)
            if self.kv_evict and past_key_values is not None:
                self.evict(past_key_values, q, k, v)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out), None

    def exact_attention(self, q, k, v, t, n):
        return F.scaled_dot_product_attention(q, k, v, is_causal=t == n and t > 1,
                                              scale=self.scaling, enable_gqa=True)

    def routes(self, q, k):
        """Per-head top-V key columns by mean softmax prob of the last n_route queries.

        K stays at KV-head width: each KV head scores all `groups` of its query heads in one
        matmul, so no repeat_interleave'd copy of K is made."""
        n = k.shape[-2]
        budget = min(vertical_budget(n, self.vertical, self.vertical_sqrt), max(0, n - self.sink - self.window))
        r = min(self.n_route, n)
        if self.route_queries == "spread":
            # Queries evenly spaced over the prompt: globally attended columns, not question-specific.
            qpos = torch.linspace(self.sink + self.window, n - 1, r, device=q.device).long()
        else:
            qpos = torch.arange(n - r, n, device=q.device)
        g, hk = self.groups, self.num_kv_heads
        qa = q[0][:, qpos, :].reshape(hk, g * r, self.head_dim)         # [Hkv, g*r, d]
        qpos_g = qpos.repeat(g)                                         # row j*r+i -> position of query i
        col = torch.zeros(self.num_heads, n, device=q.device, dtype=torch.float32)
        keypos = torch.arange(n, device=q.device)
        step = max(1, 8 // g)                                           # ~8 query heads per chunk
        for k0 in range(0, hk, step):
            k1 = min(k0 + step, hk)
            s = torch.matmul(qa[k0:k1], k[0, k0:k1].transpose(-1, -2)).float() * self.scaling
            s.masked_fill_(keypos[None, None, :] > qpos_g[None, :, None], -float("inf"))
            p = s.softmax(-1).view(k1 - k0, g, r, n)
            pooled = p.amax(2) if self.route_pool == "max" else p.mean(2)
            col[k0 * g:k1 * g] = pooled.reshape((k1 - k0) * g, n)
        col[:, :self.sink] = -1
        col[:, n - self.window:] = -1                                   # handled by the window branch
        if self.kv_evict:
            self._col = col
        idx = col.topk(budget, dim=-1).indices.sort(-1).values.to(torch.int32)
        return idx

    def sparse_prefill(self, q, k, v):
        n = k.shape[-2]
        idx = self.routes(q, k)
        self.last_routes = idx
        bh = self.num_heads
        # Q/K/V go to the kernel as strided views (K/V at KV-head width, GQA resolved in-kernel):
        # no pre-scaled Q copy and no repeat_interleave'd K/V.
        Q, K, V = q[0], k[0], v[0]
        flash = dict(num_heads=self.num_heads, kv_groups=self.groups, scale=self.scaling,
                     block_m=self.block_m, block_n=self.block_n,
                     num_warps=self.num_warps, num_stages=self.num_stages)
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
                                token_mask=None, route_chunk=chunk, **flash)[None]
        r = min(self.exact_tail, n)
        if r:
            # Final r prompt queries (question / answer cue) attend exactly: O(r * n). Same kernel
            # with window=n (dense causal row), so no [H, r, n] score/mask workspace.
            none = torch.full((bh, 1, 1), -1, dtype=torch.int32, device=q.device)
            out[:, :, n - r:] = bigger_bird_flash(Q[:, n - r:], K, V, none, front=0, window=n,
                                                  token_mask=None, route_chunk=None, **flash)[None]
        return out


    def evict_scores(self, q, k):
        """[Hkv, n] key importance from the last `evict_obs` queries: exact softmax, mean (or with `evict_obs_max`,
        max) over those queries, max over each KV head's query group. Max keeps a key that any single query
        (e.g. one key name in a multi-part question) attends to strongly."""
        n, g, hk = k.shape[-2], self.groups, self.num_kv_heads
        r = min(self.evict_obs, n)
        qa = q[0][:, n - r:, :].reshape(hk, g * r, self.head_dim)
        qpos = torch.arange(n - r, n, device=q.device).repeat(g)
        keypos = torch.arange(n, device=q.device)
        out = torch.empty(hk, n, device=q.device, dtype=torch.float32)
        for h in range(hk):                                             # one KV head at a time: [g*r, n] fp32
            s = torch.matmul(qa[h], k[0, h].transpose(-1, -2)).float() * self.scaling
            s.masked_fill_(keypos[None, :] > qpos[:, None], -float("inf"))
            p = s.softmax(-1).view(g, r, n)
            out[h] = (p.amax(1) if self.evict_obs_max else p.mean(1)).amax(0)
        return out

    def evict(self, cache, q, k, v):
        """Rewrite this layer's cache to sink + routed + window keys per KV head (see module doc).

        Tuning (defaults reproduce the original policy): `evict_full_layers` leaves the first layers
        uncompressed; `evict_keep` sets kept middle slots per KV head (0 = g * budget); `evict_obs` > 0
        scores keys by the exact attention of the last `evict_obs` queries instead of the routing scores;
        `evict_window` keeps a longer recent window in the cache than prefill used (0 = `window`);
        `evict_pool` = p also keeps the p - 1 keys following each high-scoring key, so a multi-token
        span (e.g. a number being copied) is kept whole; `evict_pool_back` = b also keeps the b keys before it."""
        n, d = k.shape[-2], k.shape[-1]
        g, hk = self.groups, self.num_kv_heads
        idx, col, self._col = self.last_routes, self._col, None
        if self.layer_idx < self.evict_full_layers:
            self.kept = torch.arange(n, device=k.device)[None].expand(hk, -1)
            return
        member = torch.zeros(self.num_heads, n, dtype=torch.bool, device=k.device)
        member.scatter_(1, idx.long(), True)
        score = self.evict_scores(q, k) if self.evict_obs else col.view(hk, g, n).amax(1)
        if self.evict_pool > 1 or self.evict_pool_back:
            score = forward_pool(score, self.evict_pool, self.evict_pool_back)
        # Every head's routed column is kept; leftover slots go to the best remaining columns.
        prio = member.view(hk, g, n).any(1).float() * 2 + score
        w = min(max(self.window, self.evict_window), n - self.sink)
        prio[:, :self.sink] = -float("inf")
        prio[:, n - w:] = -float("inf")
        extra = min(max(self.evict_keep, g * idx.shape[-1]) if self.evict_keep else g * idx.shape[-1],
                    n - self.sink - w)
        sel = prio.topk(extra, dim=-1).indices.sort(-1).values
        ar = torch.arange(n, device=k.device)
        pos = torch.cat([ar[None, :self.sink].expand(hk, -1), sel, ar[None, n - w:].expand(hk, -1)], 1)
        gi = pos[None, :, :, None].expand(1, hk, -1, d)
        layer = cache.layers[self.layer_idx]
        layer.keys, layer.values = k.gather(2, gi), v.gather(2, gi)
        self.kept = pos


def install(model, arm, dense_layers=(), **kw):
    """Patch every decoder layer; `dense_layers` stay exact (ablation only)."""
    for i, layer in enumerate(model.model.layers):
        base = layer.self_attn
        base = getattr(base, "_rd_base", base)
        new = RDAttention(base, dense=(arm == "dense_rd" or i in dense_layers), **kw)
        new._rd_base = base
        layer.self_attn = new
    return model
