"""INT8 KV cache (symmetric, per token per KV head) as a drop-in HF cache layer.

Prefill attention still sees full-precision K/V (update() returns the inputs it was given); only
the stored copy is quantised. Cached decode dequantises one layer at a time (one Triton kernel on
CUDA, kernels/kv_int8_triton.py), so the transient is a single layer's cache; new tokens are
written into spare capacity instead of re-concatenating the cache every step. Storage is
1 byte/element plus one fp16 scale per head_dim elements (~0.51x of BF16). Composes with `kv_evict`
(modal_rd.attention): eviction assigns `layer.keys` / `layer.values`, which re-quantises the kept
rows from the original full-precision tensors.
"""
import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer

MIN_SCALE = 1e-4
GROW = 256      # decode appends into spare capacity, reallocating every GROW tokens instead of every token


def quantize(x):
    scale = (x.float().abs().amax(-1, keepdim=True) / 127).clamp_min(MIN_SCALE)
    q = (x.float() / scale).round().clamp_(-127, 127).to(torch.int8)
    return q, scale.to(torch.float16)


def dequantize(q, scale, dtype):
    return (q.float() * scale.float()).to(dtype)


try:
    from kernels import kv_int8_triton as _triton
except ImportError:  # no Triton (CPU/MPS dev box)
    _triton = None


def _use_triton(x):
    # Bit-identical Triton kernels on CUDA: one launch each instead of ~10 eager ops, and the decode-time
    # dequantise of a 65K-token layer drops from ~1.1 ms (eager fp32) to ~0.06 ms.
    return _triton is not None and x.is_cuda and x.shape[-1] & (x.shape[-1] - 1) == 0


def _quantize(x):
    return _triton.quantize(x, MIN_SCALE) if _use_triton(x) else quantize(x)


def _dequantize(q, s, dtype):
    return _triton.dequantize(q, s, dtype) if _use_triton(q) else dequantize(q, s, dtype)


class Int8KVLayer(DynamicLayer):
    """Buffers are token-major [capacity, Hkv, d] (scales [capacity, Hkv, 1]) so the valid rows [:n] are one
    contiguous block: dequantising them is a single bandwidth-bound kernel, and decode appends a row into
    spare capacity. Batch size 1 only."""

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.qk = self.qv = self.sk = self.sv = None
        self.n = 0
        self.is_initialized = True

    @staticmethod
    def _quantize(x):                                 # [1, Hkv, t, d] -> [t, Hkv, d] int8, [t, Hkv, 1] fp16
        if x.shape[0] != 1:
            raise NotImplementedError("Int8KVLayer is batch-size-1 only")
        return _quantize(x[0].transpose(0, 1))

    def _get(self, q, s):                             # -> [1, Hkv, n, d] view of the dequantised rows
        if q is None:
            return None
        return _dequantize(q[:self.n], s[:self.n], self.dtype).transpose(0, 1)[None]

    @property
    def keys(self):
        return self._get(getattr(self, "qk", None), getattr(self, "sk", None))

    @keys.setter
    def keys(self, x):
        if x is not None and x.numel():
            self.qk, self.sk = self._quantize(x)
            self.n = x.shape[-2]

    @property
    def values(self):
        return self._get(getattr(self, "qv", None), getattr(self, "sv", None))

    @values.setter
    def values(self, x):
        if x is not None and x.numel():
            self.qv, self.sv = self._quantize(x)
            self.n = x.shape[-2]

    def _reserve(self, t):
        if self.n + t <= self.qk.shape[0]:
            return
        grow = lambda x: torch.cat([x[:self.n], x.new_empty(max(t, GROW), *x.shape[1:])])
        self.qk, self.sk, self.qv, self.sv = map(grow, (self.qk, self.sk, self.qv, self.sv))

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        t = key_states.shape[-2]
        if self.qk is None:                       # prefill: store quantised, attend in full precision
            (self.qk, self.sk), (self.qv, self.sv) = self._quantize(key_states), self._quantize(value_states)
            self.n = t
            return key_states, value_states
        self._reserve(t)
        rows = slice(self.n, self.n + t)
        for x, q, s in ((key_states, self.qk, self.sk), (value_states, self.qv, self.sv)):
            if _use_triton(x):                    # quantise straight into the spare rows
                _triton.quantize(x[0].transpose(0, 1), MIN_SCALE, out=(q[rows], s[rows]))
            else:
                q[rows], s[rows] = self._quantize(x)
        self.n += t
        return self.keys, self.values

    def get_seq_length(self):
        return 0 if not self.is_initialized or self.qk is None else self.n

    def crop(self, tokens_to_remove):
        if tokens_to_remove < 0 and self.get_seq_length():
            self.n = max(0, self.n + tokens_to_remove)
        elif tokens_to_remove:
            raise NotImplementedError("Int8KVLayer.crop supports negative token counts only")

    def batch_repeat_interleave(self, repeats):
        raise NotImplementedError("Int8KVLayer is batch-size-1 only")

    batch_select_indices = batch_repeat_interleave

    def nbytes(self):
        """Bytes of the valid rows (spare decode capacity, at most GROW rows, is not counted)."""
        if self.qk is None:
            return 0
        return sum(t[:self.n].numel() * t.element_size() for t in (self.qk, self.sk, self.qv, self.sv))


def make_int8_cache(config):
    cache = DynamicCache(config=config)
    cache.layers = [Int8KVLayer() for _ in range(config.get_text_config(decoder=True).num_hidden_layers)]
    return cache
