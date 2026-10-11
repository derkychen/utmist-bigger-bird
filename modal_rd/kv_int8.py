"""INT8 KV cache (symmetric, per token per KV head) as a drop-in HF cache layer.

Prefill attention still sees full-precision K/V (update() returns the inputs it was given); only
the stored copy is quantised. Cached decode dequantises one layer at a time, so the transient is a
single layer's cache. Storage is 1 byte/element plus one fp16 scale per head_dim elements
(~0.51x of BF16). Composes with `kv_evict` (modal_rd.attention): eviction assigns `layer.keys` /
`layer.values`, which re-quantises the kept rows from the original full-precision tensors.
"""
import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer

MIN_SCALE = 1e-4


def quantize(x):
    scale = (x.float().abs().amax(-1, keepdim=True) / 127).clamp_min(MIN_SCALE)
    q = (x.float() / scale).round().clamp_(-127, 127).to(torch.int8)
    return q, scale.to(torch.float16)


def dequantize(q, scale, dtype):
    return (q.float() * scale.float()).to(dtype)


class Int8KVLayer(DynamicLayer):
    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.qk = self.qv = self.sk = self.sv = None
        self.is_initialized = True

    @property
    def keys(self):
        return None if getattr(self, "qk", None) is None else dequantize(self.qk, self.sk, self.dtype)

    @keys.setter
    def keys(self, x):
        if x is not None and x.numel():
            self.qk, self.sk = quantize(x)

    @property
    def values(self):
        return None if getattr(self, "qv", None) is None else dequantize(self.qv, self.sv, self.dtype)

    @values.setter
    def values(self, x):
        if x is not None and x.numel():
            self.qv, self.sv = quantize(x)

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        qk, sk = quantize(key_states)
        qv, sv = quantize(value_states)
        if self.qk is None:                       # prefill: store quantised, attend in full precision
            self.qk, self.sk, self.qv, self.sv = qk, sk, qv, sv
            return key_states, value_states
        self.qk, self.sk = torch.cat([self.qk, qk], -2), torch.cat([self.sk, sk], -2)
        self.qv, self.sv = torch.cat([self.qv, qv], -2), torch.cat([self.sv, sv], -2)
        return dequantize(self.qk, self.sk, self.dtype), dequantize(self.qv, self.sv, self.dtype)

    def get_seq_length(self):
        return 0 if not self.is_initialized or self.qk is None else self.qk.shape[-2]

    def crop(self, tokens_to_remove):
        if tokens_to_remove < 0 and self.get_seq_length():
            self.qk, self.sk = self.qk[..., :tokens_to_remove, :], self.sk[..., :tokens_to_remove, :]
            self.qv, self.sv = self.qv[..., :tokens_to_remove, :], self.sv[..., :tokens_to_remove, :]
        elif tokens_to_remove:
            raise NotImplementedError("Int8KVLayer.crop supports negative token counts only")

    def batch_repeat_interleave(self, repeats):
        raise NotImplementedError("Int8KVLayer is batch-size-1 only")

    batch_select_indices = batch_repeat_interleave

    def nbytes(self):
        return 0 if self.qk is None else sum(t.numel() * t.element_size() for t in (self.qk, self.sk, self.qv, self.sv))


def make_int8_cache(config):
    cache = DynamicCache(config=config)
    cache.layers = [Int8KVLayer() for _ in range(config.get_text_config(decoder=True).num_hidden_layers)]
    return cache
