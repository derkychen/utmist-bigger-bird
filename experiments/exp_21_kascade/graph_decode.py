"""CUDA-graph greedy decode for Llama-family models: one captured graph per generated token.

After an ordinary (eager) prefill into a DynamicCache, `StaticDecoder` copies the cache into fixed-capacity
per-layer K/V buffers and runs a hand-written one-token step (embed -> layers -> argmax) whose shapes never
change. The cache length, the cache write slot and the RoPE position live in GPU tensors, so the step is
captured once as a CUDA graph and replayed for every token without Python or per-kernel launch overhead.

Attention per layer follows what is installed on the model:
  * exp 21 sparse decode (`sparse_decode.SparseDecodeAttention`, pooled routes): anchor layers score every
    cached key and take the top-k outside sink / window (`kernels/bigger_bird_route.py`); other layers reuse
    them (head map rows applied); attention runs in `kernels/bigger_bird_decode.py`; both read the length
    on the GPU.
  * anything else (dense, v3, v3 + KV eviction): dense flash-decoding over the filled part of the buffer
    (same kernel, front=0, window=capacity, no routed keys).
KV eviction only shrinks the cache at prefill, so it works here too: the cache slot (cache length) and the RoPE
position (absolute token index) are tracked separately.

  dec = StaticDecoder(model, capacity=prompt_len + max_new)
  dec.load(cache, prompt_len, first_token)   # after prefill
  tokens = dec.generate(max_new, graph=True)
"""
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from experiments.exp_21_kascade.sparse_decode import SparseDecodeAttention
from kernels.bigger_bird_decode import bigger_bird_decode
from kernels.bigger_bird_route import bigger_bird_route


class StaticDecoder:
    def __init__(self, model, capacity):
        cfg = model.config
        self.model, self.capacity = model, capacity
        self.layers = model.model.layers
        self.hk, self.h = cfg.num_key_value_heads, cfg.num_attention_heads
        self.g = self.h // self.hk
        attn0 = self.layers[0].self_attn
        self.d = getattr(attn0, "head_dim", cfg.hidden_size // self.h)
        dev, dt = model.device, model.dtype if model.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        self.k = [torch.zeros(self.hk, capacity, self.d, device=dev, dtype=dt) for _ in self.layers]
        self.v = [torch.zeros_like(x) for x in self.k]
        self.length = torch.zeros(1, dtype=torch.long, device=dev)      # filled cache slots
        self.position = torch.zeros(1, dtype=torch.long, device=dev)    # RoPE position of the next token
        self.token = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.step_i = torch.zeros(1, dtype=torch.long, device=dev)
        self.out = torch.zeros(capacity, dtype=torch.long, device=dev)
        self.slots = torch.arange(capacity, device=dev)
        self.no_routes = torch.empty(self.hk, 0, dtype=torch.long, device=dev)
        self.graph = None
        self.sparse = [isinstance(layer.self_attn, SparseDecodeAttention) for layer in self.layers]
        for layer, sp in zip(self.layers, self.sparse):
            if sp and not layer.self_attn.pool:
                raise ValueError("StaticDecoder supports pooled exp 21 routes only")

    def load(self, cache, prompt_len, first_token):
        """Copy a prefilled DynamicCache into the buffers. `prompt_len` = RoPE position of `first_token`."""
        m = cache.layers[0].keys.shape[-2]
        if m >= self.capacity:
            raise ValueError("capacity too small")
        for i, layer in enumerate(cache.layers):
            if layer.keys.shape[-2] != m:
                raise ValueError("all layers must hold the same number of cached keys")
            self.k[i][:, :m].copy_(layer.keys[0])
            self.v[i][:, :m].copy_(layer.values[0])
        self.length.fill_(m)
        self.position.fill_(prompt_len)
        self.token.fill_(first_token)
        self.step_i.zero_()

    def attention(self, i, attn, q, k, v):
        """q [H, d]; this layer's new k/v [Hkv, d] already written. Returns [H, d]."""
        K, V = self.k[i], self.v[i]
        if not self.sparse[i]:
            return bigger_bird_decode(q, K, V, self.no_routes, front=0, window=self.capacity, scale=attn.scaling,
                                      n=self.length)
        if attn.anchor:
            attn.shared["idx"] = bigger_bird_route(q, K, attn.decode_vertical, front=attn.sink,
                                                   window=attn.decode_window, scale=attn.scaling, n=self.length)
        idx = attn.shared["idx"] if attn.rows is None else attn.shared["idx"][attn.rows]
        return bigger_bird_decode(q, K, V, idx, front=attn.sink, window=attn.decode_window, scale=attn.scaling,
                                  n=self.length)

    def step(self):
        """One greedy token; every shape is fixed and all state lives in GPU tensors (graph-capturable)."""
        h = self.model.model.embed_tokens(self.token)                                       # [1, 1, hidden]
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
            self.k[i].index_copy_(1, slot, k[0])
            self.v[i].index_copy_(1, slot, v[0])
            o = self.attention(i, attn, q[0, :, 0], k, v)
            h = h + base.o_proj(o.reshape(1, 1, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        logits = self.model.lm_head(self.model.model.norm(h))
        nxt = logits[0, -1].argmax().view(1, 1)
        self.out.index_copy_(0, self.step_i, nxt[0])
        self.token.copy_(nxt)
        self.position.add_(1)
        self.step_i.add_(1)

    @torch.inference_mode()
    def capture(self):
        """Record `step` as a CUDA graph (state is saved and restored around the warm-up runs)."""
        saved = [t.clone() for t in (self.length, self.position, self.token, self.step_i)]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):                    # compile Triton kernels / warm cuBLAS outside the capture
                self.step()
                for t, x in zip((self.length, self.position, self.token, self.step_i), saved):
                    t.copy_(x)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.step()
        for t, x in zip((self.length, self.position, self.token, self.step_i), saved):
            t.copy_(x)

    @torch.inference_mode()
    def run(self, steps, graph=True):
        """Advance `steps` tokens (replaying the graph or eagerly). No host sync inside the loop."""
        if graph and self.graph is None:
            self.capture()
        for _ in range(steps):
            self.graph.replay() if graph else self.step()

    def rewind(self, r):
        """Drop the last `r` consumed tokens (e.g. steps run past an EOS)."""
        self.length.sub_(r)
        self.position.sub_(r)

    def feed(self, token):
        """Force the next input token (e.g. an appended prompt turn); the following `run(1)` consumes it."""
        self.token.fill_(token)

    def generated(self):
        """Tokens produced so far (the argmax after each step), as a Python list."""
        return self.out[:int(self.step_i)].tolist()
