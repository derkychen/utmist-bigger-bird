"""Exp 21 phase 2b — sparse decode with Kascade-style anchor-layer reuse.

Prefill is unchanged Bigger Bird v3. During cached decode (one new query, n cached
keys), each layer attends to sink + the most recent `window` keys + `vertical`
routed keys per head instead of the whole cache:
  * anchor layers score the current query against every cached key and take the
    per-head top-`vertical` outside sink/window;
  * non-anchor layers reuse the nearest earlier anchor's indices from this step,
    head for head or through a Kascade head map (`model.head_map`).
Routes are recomputed every decode step at the anchors, so the routed set never
overlaps the (sliding) window and no key is counted twice. `decode_window` / `decode_vertical`
set the decode window and routed budget separately from prefill (default: the same).

Appending t > 1 tokens to a cache (t < n) uses unmasked attention, as in v3, so feed
multi-token turns one token at a time (see eval_decode.py).

The attention itself runs in `kernels/bigger_bird_decode.py` (split-K, keys read in place) and pooled
anchor routing in `kernels/bigger_bird_route.py` (scores read each key once) on CUDA FP16/BF16;
BB_DECODE_KERNEL=0 selects the original gather + fp32 PyTorch path.

With `pool=True` (Kascade's GQA pooling) the anchor averages the softmax
probabilities of the query heads sharing a KV head and picks one set per KV head;
those heads then read the same gathered keys/values.

KV reads per layer per generated token, per KV head (K and V counted separately):
dense 2n; anchor n (scoring K) + 2S; reuser 2S, with S = sink + window + P and
P = vertical when pooled, or up to groups * vertical distinct picks when each
query head picks its own (upper bound).
"""
import os

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from kernels.bigger_bird_decode import bigger_bird_decode
from kernels.bigger_bird_route import bigger_bird_route
from modal_rd import attention as rd

KERNEL = os.environ.get("BB_DECODE_KERNEL", "1") != "0"


def expand_spans(idx, span, n, window):
    """Span pooling: each picked position p -> p .. p + span - 1 (like KV eviction's `evict_pool`), so multi-token
    items such as numbers are attended whole. idx [R, K] -> [R, K * span]; positions reaching the decode window
    (>= n - window, already attended) and duplicates from overlapping spans become -1 (ignored by the kernels).
    `n` may be an int or a 1-element GPU tensor (CUDA-graph safe)."""
    if span <= 1:
        return idx
    e = (idx[..., None] + torch.arange(span, device=idx.device)).flatten(-2)
    e = e.masked_fill(e >= n - window, -1).sort(-1).values
    dup = torch.zeros_like(e, dtype=torch.bool)
    dup[..., 1:] = e[..., 1:] == e[..., :-1]
    return e.masked_fill(dup, -1)


class SparseDecodeAttention(rd.RDAttention):
    def __init__(self, base, *, anchor, shared, rows=None, pool=False, decode_window=None, decode_vertical=None,
                 decode_span=1, **kw):
        super().__init__(base, **kw)
        self.anchor, self.shared, self.rows, self.pool = anchor, shared, rows, pool
        self.decode_window = decode_window or self.window
        self.decode_vertical = decode_vertical or self.vertical
        self.decode_span = decode_span or 1      # routed slots = decode_vertical: decode_vertical // span picks x span

    @property
    def picks(self):
        return max(1, self.decode_vertical // self.decode_span)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kwargs):
        b, t, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        if t == 1 and type(past_key_values).__name__ == "OffloadKVCache":    # offload_cache.py (1M validation)
            from experiments.exp_21_kascade.offload_cache import offload_decode
            out = offload_decode(self, q, k, v, past_key_values.layers[self.layer_idx])
            return self.o_proj(out.transpose(1, 2).reshape(b, t, -1)), None
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        n = k.shape[-2]
        if b != 1:
            raise ValueError("batch size 1 only")
        if t == 1 and n > self.sink + self.decode_window + self.decode_vertical:
            out = self.sparse_decode(q, k, v)
        elif self.dense or t < n or n <= self.sink + self.window:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=t == n and t > 1,
                                                 scale=self.scaling, enable_gqa=True)
        else:
            out = self.sparse_prefill(q, k, v)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out), None

    @staticmethod
    def use_kernel(q):
        return KERNEL and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)

    def kernel(self, q, k, v, idx):
        """idx [Hkv, R] (pooled) or [H, R]; returns [1, H, 1, d]."""
        out = bigger_bird_decode(q[0, :, 0], k[0], v[0], idx, front=self.sink, window=self.decode_window,
                                 scale=self.scaling)
        return out[None, :, None]

    def sparse_decode(self, q, k, v):
        if self.pool:
            return self.sparse_decode_pooled(q, k, v)
        n, H, G = k.shape[-2], self.num_heads, self.groups
        if self.anchor:
            # [KV, G, d] x [KV, d, n] -> per-head scores without expanding K to all heads
            s = torch.matmul(q[0, :, 0].view(self.num_kv_heads, G, -1), k[0].transpose(-1, -2)).view(H, n)
            s[:, :self.sink] = -float("inf")
            s[:, n - self.decode_window:] = -float("inf")
            self.shared["idx"] = expand_spans(s.topk(self.picks, dim=-1, sorted=False).indices,
                                              self.decode_span, n, self.decode_window)
            self.shared["anchor_layer"] = self.layer_idx
        dev = q.device
        idx = self.shared["idx"] if self.rows is None else self.shared["idx"][self.rows]
        if self.use_kernel(q):
            return self.kernel(q, k, v, idx)
        if self.decode_span > 1:
            raise ValueError("span pooling needs the decode kernel (CUDA FP16/BF16)")
        pos = torch.cat([torch.arange(self.sink, device=dev).expand(H, -1),
                         torch.arange(n - self.decode_window, n, device=dev).expand(H, -1),
                         idx], dim=1)                                       # [H, S]
        kv_head = (torch.arange(H, device=dev) // G)[:, None]
        Ks, Vs = k[0][kv_head, pos], v[0][kv_head, pos]                     # [H, S, d]
        # fp32 scores/output: bf16 matmul rounding flips near-tie tokens on models with large logits (Qwen2)
        att = (torch.matmul(q[0].float(), Ks.float().transpose(-1, -2)) * self.scaling).softmax(-1)
        return torch.matmul(att, Vs.float()).to(q.dtype)[None]              # [1, H, 1, d]

    def sparse_decode_pooled(self, q, k, v):
        n, KV, G = k.shape[-2], self.num_kv_heads, self.groups
        qg = q[0, :, 0].view(KV, G, -1)                                     # [KV, G, d]
        if self.anchor and self.use_kernel(q):
            self.shared["idx"] = expand_spans(
                bigger_bird_route(q[0, :, 0], k[0], self.picks, front=self.sink, window=self.decode_window,
                                  scale=self.scaling), self.decode_span, n, self.decode_window)
        elif self.anchor:
            p = (torch.matmul(qg, k[0].transpose(-1, -2)) * self.scaling).float().softmax(-1).mean(1)  # [KV, n]
            p[:, :self.sink] = -1
            p[:, n - self.decode_window:] = -1
            self.shared["idx"] = expand_spans(p.topk(self.picks, dim=-1, sorted=False).indices,
                                              self.decode_span, n, self.decode_window)
        dev = q.device
        idx = self.shared["idx"] if self.rows is None else self.shared["idx"][self.rows]
        if self.use_kernel(q):
            return self.kernel(q, k, v, idx)
        if self.decode_span > 1:
            raise ValueError("span pooling needs the decode kernel (CUDA FP16/BF16)")
        pos = torch.cat([torch.arange(self.sink, device=dev).expand(KV, -1),
                         torch.arange(n - self.decode_window, n, device=dev).expand(KV, -1),
                         idx], dim=1)                                       # [KV, S]
        heads = torch.arange(KV, device=dev)[:, None]
        Ks, Vs = k[0][heads, pos], v[0][heads, pos]                         # [KV, S, d]
        att = (torch.matmul(qg.float(), Ks.float().transpose(-1, -2)) * self.scaling).softmax(-1)  # [KV, G, S]
        out = torch.matmul(att, Vs.float()).to(q.dtype)                     # [KV, G, d]
        return out.reshape(1, self.num_heads, 1, -1)


def install(model, anchors, head_map=None, pool=False, decode_window=None, decode_vertical=None, decode_span=1,
            **attn):
    """v3 prefill everywhere; sparse decode with routes from `anchors` (others reuse)."""
    anchors = set(anchors)
    if 0 not in anchors:
        raise ValueError("layer 0 must be an anchor")
    G = model.config.num_attention_heads // model.config.num_key_value_heads
    shared = {}
    for i, layer in enumerate(model.model.layers):
        base = getattr(layer.self_attn, "_rd_base", layer.self_attn)
        rows = None
        if head_map is not None and i not in anchors:
            # pooled: one row per KV head (a KV group's heads share one mapped anchor group)
            rows = torch.as_tensor(head_map[i][::G] // G if pool else head_map[i], device=model.device)
        new = SparseDecodeAttention(base, anchor=i in anchors, shared=shared, rows=rows, pool=pool,
                                    decode_window=decode_window, decode_vertical=decode_vertical,
                                    decode_span=decode_span, **attn)
        new._rd_base = base
        layer.self_attn = new
    return model


def kv_reads(n, num_layers, anchors, sink, window, vertical, spread=1):
    """KV tokens read per KV head per generated token (all layers), sparse vs dense.

    spread = 1 when picks are pooled per KV head; = groups (upper bound) when each
    query head picks its own columns.
    """
    if n <= sink + window + vertical:
        return 2 * n * num_layers, 2 * n * num_layers
    S = sink + window + min(spread * vertical, n - sink - window)
    a = len(anchors)
    return a * (n + 2 * S) + (num_layers - a) * 2 * S, 2 * n * num_layers
