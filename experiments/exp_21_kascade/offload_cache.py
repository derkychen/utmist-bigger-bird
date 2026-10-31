"""CPU-offloaded KV cache for exp 21 sparse decode (validation at ~1M tokens on one H100; exact, slow).

Every layer's K/V lives in preallocated CPU buffers; only the anchor layers' keys are copied back to the GPU
after prefill (~10 GB for 5 anchors at 1M tokens), because anchors score every cached key each step. Each
decode step then, per layer:
  1. writes the new token's K/V to the CPU buffers (and the anchor's GPU keys),
  2. anchors pick routes on the GPU (`kernels/bigger_bird_route.py`), other layers reuse them as usual,
  3. gathers sink + window + routed keys/values from CPU memory and copies them to the GPU,
  4. attends over that compact set (sink | routed | window) with `kernels/bigger_bird_decode.py`, which visits
     the keys in the same order as the on-GPU sparse decode, so the result matches it.
Prefill is plain v3 (use exp 21 with `decodeonly`); each layer's K/V goes to the CPU as soon as it is computed,
so the full cache never sits on the GPU. Pooled routes only, batch 1, one token per decode step.
"""
import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer

from kernels.bigger_bird_decode import bigger_bird_decode
from kernels.bigger_bird_route import bigger_bird_route


class OffloadLayer(DynamicLayer):
    def __init__(self, hk, d, capacity, dtype, anchor):
        super().__init__()
        self.kc = torch.empty(hk, capacity, d, dtype=dtype)          # CPU
        self.vc = torch.empty(hk, capacity, d, dtype=dtype)
        self.anchor, self.n, self.kg = anchor, 0, None               # kg: anchor keys on the GPU after prefill
        self.dtype, self.device, self.is_initialized = dtype, torch.device("cpu"), True
        self.keys = self.values = torch.empty(0)

    def get_seq_length(self):
        return self.n

    def update(self, key_states, value_states, *args, **kwargs):
        """Store new K/V on the CPU; return them unchanged (prefill attends over exactly these)."""
        t = key_states.shape[-2]
        if self.n + t > self.kc.shape[1]:
            raise ValueError("offload cache capacity exceeded")
        self.kc[:, self.n:self.n + t].copy_(key_states[0])
        self.vc[:, self.n:self.n + t].copy_(value_states[0])
        if self.kg is not None:
            self.kg[:, self.n:self.n + t].copy_(key_states[0])
        self.n += t
        return key_states, value_states


class OffloadKVCache(DynamicCache):
    def __init__(self, model, capacity):
        super().__init__(config=model.config)
        cfg = model.config
        hk, d = cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
        dtype = model.lm_head.weight.dtype
        self.layers = [OffloadLayer(hk, d, capacity, dtype, getattr(layer.self_attn, "anchor", False))
                       for layer in model.model.layers]

    def after_prefill(self, device="cuda"):
        """Copy the anchor layers' keys to the GPU (they score every cached key at each decode step)."""
        for layer in self.layers:
            if layer.anchor:
                layer.kg = torch.empty(layer.kc.shape, dtype=layer.kc.dtype, device=device)
                layer.kg[:, :layer.n].copy_(layer.kc[:, :layer.n])


def offload_decode(attn, q, k, v, layer):
    """One decode step of exp 21 sparse decode (pooled routes) over an OffloadLayer. q [1, H, 1, d]."""
    if not attn.pool:
        raise ValueError("offloaded decode supports pooled routes only")
    layer.update(k, v)
    n, dw = layer.n, attn.decode_window
    qh = q[0, :, 0]
    if attn.anchor:
        from experiments.exp_21_kascade.sparse_decode import expand_spans
        attn.shared["idx"] = expand_spans(
            bigger_bird_route(qh, layer.kg[:, :n], attn.picks, front=attn.sink, window=dw, scale=attn.scaling),
            attn.decode_span, n, dw)
    idx = attn.shared["idx"] if attn.rows is None else attn.shared["idx"][attn.rows]      # -1 = unused slot
    hk, r = idx.shape
    dev = idx.device
    # compact layout sink | routed | window, so the kernel's window is the last `dw` keys as on the GPU path
    pos = torch.cat([torch.arange(attn.sink, device=dev).expand(hk, -1), idx.clamp(min=0),
                     torch.arange(n - dw, n, device=dev).expand(hk, -1)], 1).cpu()                 # [Hkv, S]
    heads = torch.arange(hk)[:, None]
    ks = layer.kc[heads, pos].to(qh.device, non_blocking=True)
    vs = layer.vc[heads, pos].to(qh.device, non_blocking=True)
    ridx = torch.where(idx >= 0, attn.sink + torch.arange(r, device=dev), -1)
    out = bigger_bird_decode(qh, ks, vs, ridx, front=attn.sink, window=dw, scale=attn.scaling)
    return out[None, :, None]
