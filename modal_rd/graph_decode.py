"""CUDA-graph greedy decode for Llama-family models (batch 1).

Prefill stays an ordinary eager forward. Decode runs a hand-written one-token step (embed -> layers ->
argmax) over fixed-capacity per-layer K/V buffers whose shapes never change; the cache length, write slot
and RoPE position live in GPU tensors. The step is captured once as a CUDA graph and replayed per token,
removing Python and per-kernel launch overhead (v3 at ~128K on an H100: 25.5 -> 15.1 ms/token, same tokens).

  dec, out, prefill_s = graph_prefill(model, input_ids, max_new)    # out.logits = last-position logits
  tokens, steps = dec.generate(max_new - 1, eos_ids)                # after the first token (argmax of out)

* Memory: without KV eviction, prefill writes straight into the decoder's buffers (no second cache copy).
  With `kv_evict`, prefill uses a normal DynamicCache (each layer is evicted right after its prefill) and the
  kept keys are copied into buffers sized for them afterwards.
* Attention per layer: dense flash-decoding over the filled buffer (`kernels/bigger_bird_decode.py`,
  front=0, window=capacity, no routes) for dense / v3 / v3 + eviction; exp 21 sparse decode layers
  (`experiments/exp_21_kascade/sparse_decode.py`, pooled routes) use fused anchor routing plus the sparse
  decode kernel. Both read the cache length on the GPU.
* `--cpu-embed` (modal_rd.cpu_embed) works: the next token's embedding row is looked up on the CPU between
  steps (one host sync and an 8 KB copy per token) instead of inside the graph.
* Compressed full cache (modal_rd/turbo_kv.py, `graph_prefill(..., turbo_bits=b)`): the decoder keeps 4-bit codes and
  norms instead of BF16 K/V, quantizes each new token in the graph and attends with kernels/turbo_decode.py. For
  long prompts the codes are staged on the CPU during prefill and the decoder is rebuilt (and re-captured) per
  prompt, so prefill activations and the GPU codes are never live together.
* Not supported: the INT8 KV cache (modal_rd.kv_int8), batch > 1, sampling.
* Generation runs in blocks of `block` steps and stops after the block that produced an EOS; the cache is
  rewound to just after the EOS so a further turn can be appended (`feed` + `run`).
"""
import time

import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from kernels.bigger_bird_decode import bigger_bird_decode
from kernels.bigger_bird_route import bigger_bird_route
from kernels.turbo_decode import turbo_decode


class BufferLayer(DynamicLayer):
    """DynamicLayer whose keys / values are views of preallocated [Hkv, capacity, d] buffers."""

    def __init__(self, kb, vb):
        super().__init__()
        self.kb, self.vb, self.n = kb, vb, 0
        self.dtype, self.device, self.is_initialized = kb.dtype, kb.device, True
        self.keys, self.values = kb[None, :, :0], vb[None, :, :0]

    def update(self, key_states, value_states, *args, **kwargs):
        t = key_states.shape[-2]
        if self.n + t > self.kb.shape[1]:
            raise ValueError("graph decode buffer too small for the prompt")
        self.kb[:, self.n:self.n + t].copy_(key_states[0])
        self.vb[:, self.n:self.n + t].copy_(value_states[0])
        self.n += t
        self.keys, self.values = self.kb[None, :, :self.n], self.vb[None, :, :self.n]
        return self.keys, self.values


def _sparse_decode_layer(attn):
    """exp 21 SparseDecodeAttention (duck-typed so modal_rd does not import the experiment)."""
    return getattr(attn, "decode_vertical", None) is not None


class StaticDecoder:
    def __init__(self, model, capacity, turbo=None):
        cfg = model.config
        self.model, self.capacity = model, capacity
        self.layers = model.model.layers
        self.hk, self.h = cfg.num_key_value_heads, cfg.num_attention_heads
        attn0 = self.layers[0].self_attn
        self.d = getattr(attn0, "head_dim", cfg.hidden_size // self.h)
        dev, dt = model.lm_head.weight.device, torch.bfloat16
        if model.lm_head.weight.dtype in (torch.float16, torch.bfloat16):
            dt = model.lm_head.weight.dtype
        self.turbo = turbo              # modal_rd.turbo_kv.TurboQuantizer: compressed cache instead of BF16 K/V
        if turbo is None:
            self.k = [torch.zeros(self.hk, capacity, self.d, device=dev, dtype=dt) for _ in self.layers]
            self.v = [torch.zeros_like(x) for x in self.k]
        else:
            codes = lambda: [torch.zeros(self.hk, capacity, self.d // 2, device=dev, dtype=torch.uint8) for _ in self.layers]
            norms = lambda: [torch.zeros(self.hk, capacity, device=dev, dtype=torch.float16) for _ in self.layers]
            self.kc, self.vc, self.kn, self.vn = codes(), codes(), norms(), norms()
        self.length = torch.zeros(1, dtype=torch.long, device=dev)      # filled cache slots
        self.position = torch.zeros(1, dtype=torch.long, device=dev)    # RoPE position of the next token
        self.token = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.step_i = torch.zeros(1, dtype=torch.long, device=dev)
        self.out = torch.zeros(capacity, dtype=torch.long, device=dev)
        self.no_routes = torch.empty(self.hk, 0, dtype=torch.long, device=dev)
        self.graph = None
        self.sparse = [_sparse_decode_layer(layer.self_attn) for layer in self.layers]
        for layer, sp in zip(self.layers, self.sparse):
            if sp and not layer.self_attn.pool:
                raise ValueError("graph decode supports pooled exp 21 routes only")
        if turbo is not None and any(self.sparse):
            raise ValueError("the compressed cache supports dense decode only")
        # CPU embedding table (modal_rd.cpu_embed): look the next token up on the host between steps
        emb = model.model.embed_tokens
        self.host_embed = getattr(emb, "weight_cpu", None)
        if self.host_embed is not None:
            self.h_in = torch.zeros(1, 1, self.host_embed.shape[1], device=dev, dtype=self.host_embed.dtype)
            self.h_stage = torch.zeros(self.host_embed.shape[1], dtype=self.host_embed.dtype).pin_memory()

    def buffer_cache(self):
        """A DynamicCache whose layers write prefill K/V straight into this decoder's buffers."""
        cache = DynamicCache(config=self.model.config)
        cache.layers = [BufferLayer(k, v) for k, v in zip(self.k, self.v)]
        return cache

    @torch.inference_mode()
    def load(self, cache, prompt_len, first_token):
        """Take a prefilled cache (a `buffer_cache()`, a DynamicCache or, with `turbo`, a TurboKVCache whose codes may
        sit on the CPU) and set the first input token."""
        if self.turbo is not None:
            m = cache.layers[0].n
            if m >= self.capacity:
                raise ValueError("graph decode buffer too small")
            for i, layer in enumerate(cache.layers):
                for dst, src in ((self.kc, layer.kc), (self.vc, layer.vc), (self.kn, layer.kn), (self.vn, layer.vn)):
                    dst[i][:, :m].copy_(src[:, :m])
            self.length.fill_(m)
            self.position.fill_(prompt_len)
            self.step_i.zero_()
            self.feed(first_token)
            return
        layers = getattr(cache, "layers", None)
        if not layers or not all(isinstance(layer, DynamicLayer) and torch.is_tensor(layer.keys) for layer in layers) \
                or layers[0].keys.dtype != self.k[0].dtype:
            raise ValueError("graph decode needs a BF16/FP16 DynamicCache (INT8 KV cache is not supported)")
        m = layers[0].keys.shape[-2]
        if m >= self.capacity:
            raise ValueError("graph decode buffer too small")
        for i, layer in enumerate(layers):
            if layer.keys.shape[-2] != m:
                raise ValueError("all layers must hold the same number of cached keys")
            if layer.keys.data_ptr() != self.k[i].data_ptr():
                self.k[i][:, :m].copy_(layer.keys[0])
                self.v[i][:, :m].copy_(layer.values[0])
        self.length.fill_(m)
        self.position.fill_(prompt_len)
        self.step_i.zero_()
        self.feed(first_token)

    @torch.inference_mode()     # the decoder's state tensors are created under inference mode
    def feed(self, token):
        """Set the next input token (prompt continuation or a generated token)."""
        self.token.fill_(token)
        if self.host_embed is not None:
            self.h_stage.copy_(self.host_embed[token])
            self.h_in.copy_(self.h_stage, non_blocking=True)

    def attention(self, i, attn, q):
        """q [H, d]; this layer's new k/v already written. Returns [H, d]."""
        if self.turbo is not None:
            return turbo_decode(q, self.kc[i], self.kn[i], self.vc[i], self.vn[i], self.turbo.c_list, self.turbo.rot,
                                scale=attn.scaling, n=self.length)
        K, V = self.k[i], self.v[i]
        if not self.sparse[i]:
            return bigger_bird_decode(q, K, V, self.no_routes, front=0, window=self.capacity, scale=attn.scaling,
                                      n=self.length)
        if attn.anchor:
            span = getattr(attn, "decode_span", 1)
            idx = bigger_bird_route(q, K, max(1, attn.decode_vertical // span), front=attn.sink,
                                    window=attn.decode_window, scale=attn.scaling, n=self.length)
            if span > 1:        # exp 21 span pooling (sparse_decode.expand_spans), fixed shape
                e = (idx[..., None] + torch.arange(span, device=idx.device)).flatten(-2)
                e = e.masked_fill(e >= self.length - attn.decode_window, -1).sort(-1).values
                dup = torch.zeros_like(e, dtype=torch.bool)
                dup[..., 1:] = e[..., 1:] == e[..., :-1]
                idx = e.masked_fill(dup, -1)
            attn.shared["idx"] = idx
        idx = attn.shared["idx"] if attn.rows is None else attn.shared["idx"][attn.rows]
        return bigger_bird_decode(q, K, V, idx, front=attn.sink, window=attn.decode_window, scale=attn.scaling,
                                  n=self.length)

    def step(self):
        """One greedy token; every shape is fixed and all state lives in GPU tensors (graph-capturable)."""
        h = self.h_in if self.host_embed is not None else self.model.model.embed_tokens(self.token)
        cos, sin = self.model.model.rotary_emb(h, self.position[None])
        slot = self.length.clone()
        self.length.add_(1)
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            base = getattr(attn, "_rd_base", attn)
            x = layer.input_layernorm(h)
            q = base.q_proj(x).view(1, 1, self.h, self.d).transpose(1, 2)
            k = base.k_proj(x).view(1, 1, self.hk, self.d).transpose(1, 2)
            v = base.v_proj(x).view(1, 1, self.hk, self.d).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if self.turbo is None:
                self.k[i].index_copy_(1, slot, k[0])
                self.v[i].index_copy_(1, slot, v[0])
            else:                               # quantize the new token into the compressed cache
                for x, codes, norms in ((k[0], self.kc[i], self.kn[i]), (v[0], self.vc[i], self.vn[i])):
                    c, nrm = self.turbo.encode(x)
                    codes.index_copy_(1, slot, c)
                    norms.index_copy_(1, slot, nrm)
            o = self.attention(i, attn, q[0, :, 0])
            h = h + base.o_proj(o.reshape(1, 1, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        logits = self.model.lm_head(self.model.model.norm(h))
        nxt = logits[0, -1].argmax().view(1, 1)
        self.out.index_copy_(0, self.step_i, nxt[0])
        self.token.copy_(nxt)
        self.position.add_(1)
        self.step_i.add_(1)

    def _state(self):
        return [t.clone() for t in (self.length, self.position, self.token, self.step_i)]

    def _restore(self, saved):
        for t, x in zip((self.length, self.position, self.token, self.step_i), saved):
            t.copy_(x)

    @torch.inference_mode()
    def capture(self):
        """Record `step` as a CUDA graph (decoder state is restored around the warm-up runs)."""
        saved = self._state()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):                    # compile Triton kernels / warm cuBLAS outside the capture
                self.step()
                self._restore(saved)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.step()
        self._restore(saved)

    @torch.inference_mode()
    def run(self, steps, graph=True):
        """Advance `steps` tokens, each consuming the current input token and producing the next."""
        if graph and self.graph is None:
            self.capture()
        for _ in range(steps):
            self.graph.replay() if graph else self.step()
            if self.host_embed is not None:
                self.feed(int(self.token))

    def generated(self):
        """All tokens produced since `load`, as a Python list."""
        return self.out[:int(self.step_i)].tolist()

    @torch.inference_mode()
    def rewind(self, r):
        """Drop the last `r` consumed tokens from the cache (e.g. steps run past an EOS)."""
        self.length.sub_(r)
        self.position.sub_(r)
        self.step_i.sub_(r)

    @torch.inference_mode()
    def generate(self, max_steps, eos=(), block=8, graph=True):
        """Greedy steps until a produced token is in `eos` (kept) or `max_steps`. Returns (tokens, steps run).

        The cache is rewound so it ends right after the input that produced the EOS, as in eager decoding
        that stops on EOS; the EOS itself becomes the next input token."""
        start, steps, toks = int(self.step_i), 0, []
        eos = set(eos)
        while steps < max_steps:
            b = min(block, max_steps - steps)
            self.run(b, graph)
            steps += b
            new = self.out[start + steps - b:start + steps].tolist()
            toks += new
            if eos.intersection(new):
                break
        cut = next((j + 1 for j, t in enumerate(toks) if t in eos), len(toks))
        if steps > cut:
            self.rewind(steps - cut)
            self.feed(toks[cut - 1])
        return toks[:cut], steps


_DECODERS = {}


def _evicting(model):
    return any(getattr(layer.self_attn, "kv_evict", False) for layer in model.model.layers)


def _key(model, turbo_bits=0):
    return tuple(id(layer.self_attn) for layer in model.model.layers) + (turbo_bits,)


def decoder_for(model, capacity):
    """Cached decoder for the installed attention modules, rebuilt if it is too small."""
    key = _key(model)
    dec = _DECODERS.get(key)
    if dec is None or dec.capacity < capacity:
        _DECODERS.clear()
        dec = None
        torch.cuda.empty_cache()
        dec = _DECODERS[key] = StaticDecoder(model, capacity)
    return dec


def _turbo_prefill(model, input_ids, max_new, headroom, position_offset, bits, key, stage_above=262144):
    """graph_prefill for the compressed full cache (see the module doc)."""
    from modal_rd.turbo_kv import TurboKVCache
    n = input_ids.shape[1]
    staged = n > stage_above
    dec = _DECODERS.get(key)
    if staged or dec is None or dec.capacity < n + max_new + 8:
        _DECODERS.clear()                       # long prompts: codes on the GPU only after prefill
        dec = None
        torch.cuda.empty_cache()
    cache = TurboKVCache(model, n + max_new + 8, bits, stage_on_cpu=staged)
    pos = torch.arange(position_offset, position_offset + n, device=input_ids.device)[None]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model(input_ids=input_ids, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0
    if dec is None:
        dec = _DECODERS[key] = StaticDecoder(model, n + max_new + 8 + (0 if staged else headroom),
                                             turbo=cache.layers[0].q)
    dec.load(cache, position_offset + n, out.logits[0, -1].argmax().item())
    del cache
    return dec, out, prefill_s


@torch.inference_mode()
def graph_prefill(model, input_ids, max_new, headroom=4096, position_offset=0, turbo_bits=0):
    """Eager prefill of `input_ids` [1, n], then load the result into a (cached) StaticDecoder.

    Returns (decoder, prefill output, prefill seconds). The first generated token is the argmax of the
    output's last logits and is already set as the decoder's next input. Capacity covers the kept cache
    plus `max_new` tokens; `headroom` extra slots let later, slightly longer prompts reuse the same graph."""
    n = input_ids.shape[1]
    key = _key(model, turbo_bits)
    if _DECODERS and key not in _DECODERS:      # another arm's buffers: free them before this prefill
        _DECODERS.clear()
        torch.cuda.empty_cache()
    if turbo_bits:
        return _turbo_prefill(model, input_ids, max_new, headroom, position_offset, turbo_bits, key)
    if _evicting(model):
        dec, cache = None, DynamicCache(config=model.config)
    else:
        dec = decoder_for(model, n + max_new + 8)
        cache = dec.buffer_cache()
    pos = torch.arange(position_offset, position_offset + n, device=input_ids.device)[None]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model(input_ids=input_ids, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0
    if dec is None:
        m = cache.layers[0].keys.shape[-2]
        dec = _DECODERS.get(key)
        if dec is None or dec.capacity < m + max_new + 8:
            dec = decoder_for(model, m + max_new + 8 + headroom)
    dec.load(cache, position_offset + n, out.logits[0, -1].argmax().item())
    del cache
    return dec, out, prefill_s


@torch.inference_mode()
def graph_generate(model, input_ids, max_new, eos=(), block=8):
    """Greedy generation like `model.generate(do_sample=False, max_new_tokens=max_new)` with a CUDA-graph decode.

    Returns (generated token ids incl. a final EOS, prefill seconds, decode seconds, decode steps run)."""
    dec, out, prefill_s = graph_prefill(model, input_ids, max_new)
    first = out.logits[0, -1].argmax().item()
    del out
    if first in set(eos) or max_new <= 1:
        return [first], prefill_s, 0.0, 0
    if dec.graph is None:
        dec.capture()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    toks, steps = dec.generate(max_new - 1, eos, block)
    torch.cuda.synchronize()
    return [first] + toks, prefill_s, time.perf_counter() - t0, steps
