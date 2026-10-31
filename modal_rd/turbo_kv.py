"""TurboQuant-style compressed KV cache (accuracy test of a full cache with no eviction).

Every cached key/value vector (head_dim d) is stored as b-bit codes, following TurboQuant's MSE quantizer
(Zandieh et al. 2025, arXiv 2504.19874): rotate by a fixed random orthogonal matrix so each coordinate is
approximately N(0, |x|^2 / d), store |x| (FP16) and quantize every coordinate of x / |x| * sqrt(d) with the
b-bit Lloyd-Max quantizer for N(0, 1). The paper's 1-bit QJL residual (unbiased inner products) is not used, so
keys are slightly biased; this is the conservative variant. Codes are kept 4 bits each (2 per byte) for
b <= 4; memory figures for b = 3 count 3 bits.

Prefill attends over the exact K/V it just computed and only stores codes. Decode attends densely over the
whole quantized cache: with `fused` (default on CUDA) through kernels/turbo_decode.py, which reads codes and norms
directly (query rotated once, output un-rotated once; ~21x faster than rebuilding the cache, RTX 3080 at 128K);
otherwise by dequantizing one layer's cache at a time. modal_rd.attention's exact decode path calls `attend`. With `stage_on_cpu`
the codes are written to CPU memory during prefill and moved to the GPU by `after_prefill()`, so they do not
add to the prefill activation peak (needed at ~1M tokens on an 80 GB GPU). That makes decode
slow (it rebuilds BF16 K/V per layer per token); it measures accuracy, not speed. Batch 1.
"""
import math

import torch
from transformers import DynamicCache

from kernels.turbo_decode import turbo_decode
from transformers.cache_utils import DynamicLayer


def lloyd_max_gaussian(bits, iters=200, grid=200001):
    """Centroids of the b-bit Lloyd-Max (MSE-optimal) scalar quantizer for N(0, 1)."""
    x = torch.linspace(-8, 8, grid, dtype=torch.float64)
    w = torch.exp(-x * x / 2)
    c = torch.linspace(-2.5, 2.5, 2 ** bits, dtype=torch.float64)
    for _ in range(iters):
        edges = (c[1:] + c[:-1]) / 2
        cell = torch.bucketize(x, edges)
        c = torch.zeros_like(c).index_add_(0, cell, w * x) / torch.zeros_like(c).index_add_(0, cell, w)
    return c.float()


class TurboQuantizer:
    def __init__(self, d, bits, device, seed=0):
        if not 1 <= bits <= 4:
            raise ValueError("bits must be in 1..4")
        g = torch.Generator().manual_seed(seed)
        q, r = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
        self.rot = (q * torch.sign(torch.diagonal(r))).float().to(device)   # random orthogonal [d, d]
        c = lloyd_max_gaussian(bits)
        c = (c - c.flip(0)) / 2                                  # exactly symmetric (kernels/turbo_decode.py relies on it)
        self.c = c.to(device)
        self.c_list = c.tolist()                                 # for the decode kernel without a GPU -> CPU read
        self.edges = (self.c[1:] + self.c[:-1]) / 2
        self.d, self.bits = d, bits

    def encode(self, x):
        """[..., d] -> (uint8 codes [..., d/2], FP16 norms [...]). Byte i holds coordinate i in its low nibble and
        coordinate i + d/2 in its high nibble (the decode kernel works on the two halves without interleaving)."""
        y = x.float() @ self.rot
        norm = y.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        codes = torch.bucketize(y / norm * math.sqrt(self.d), self.edges).to(torch.uint8)
        h = self.d // 2
        return codes[..., :h] | (codes[..., h:] << 4), norm[..., 0].half()

    def decode(self, packed, norm, dtype):
        codes = torch.cat([packed & 15, packed >> 4], -1).long()
        y = self.c[codes] * (norm.float()[..., None] / math.sqrt(self.d))
        return (y @ self.rot.T).to(dtype)


class TurboLayer(DynamicLayer):
    def __init__(self, quant, hk, d, capacity, dtype, device, chunk=65536, stage_on_cpu=False, fused=True):
        super().__init__()
        self.q, self.chunk, self.n, self.fused = quant, chunk, 0, fused
        store = "cpu" if stage_on_cpu else device
        self.kc = torch.empty(hk, capacity, d // 2, dtype=torch.uint8, device=store)
        self.vc = torch.empty_like(self.kc)
        self.kn = torch.empty(hk, capacity, dtype=torch.float16, device=store)
        self.vn = torch.empty_like(self.kn)
        self.dtype, self.device, self.is_initialized = dtype, device, True
        self.keys = self.values = torch.empty(0, device=device)

    def get_seq_length(self):
        return self.n

    def _store(self, x, codes, norms):
        for s in range(0, x.shape[1], self.chunk):           # bound the FP32 temporaries
            c, nrm = self.q.encode(x[:, s:s + self.chunk])
            codes[:, self.n + s:self.n + s + c.shape[1]] = c.to(codes.device)
            norms[:, self.n + s:self.n + s + c.shape[1]] = nrm.to(norms.device)

    def _load(self, codes, norms):
        out = torch.empty(codes.shape[0], self.n, self.q.d, dtype=self.dtype, device=self.device)
        for s in range(0, self.n, self.chunk):
            out[:, s:s + self.chunk] = self.q.decode(codes[:, s:s + self.chunk], norms[:, s:s + self.chunk], self.dtype)
        return out

    def update(self, key_states, value_states, *args, **kwargs):
        t = key_states.shape[-2]
        if self.n + t > self.kc.shape[1]:
            raise ValueError("compressed cache capacity exceeded")
        first = self.n == 0
        self._store(key_states[0], self.kc, self.kn)
        self._store(value_states[0], self.vc, self.vn)
        self.n += t
        if first:                                              # prefill: attend over the exact K/V
            return key_states, value_states
        if self.fused and t == 1 and self.kc.is_cuda:
            handle = TurboHandle(self, (1, self.kc.shape[0], self.n, self.q.d))
            return handle, handle
        return self._load(self.kc[:, :self.n], self.kn[:, :self.n])[None], \
            self._load(self.vc[:, :self.n], self.vn[:, :self.n])[None]


class TurboHandle:
    """Stands in for the K / V tensors of one decode step; modal_rd.attention calls `attend` instead of SDPA."""

    def __init__(self, layer, shape):
        self.turbo, self.shape = layer, shape

    def attend(self, q, scale):
        """q [1, H, 1, d] -> [1, H, 1, d] attention over the layer's whole compressed cache."""
        layer = self.turbo
        n = layer.n
        out = turbo_decode(q[0, :, 0], layer.kc[:, :n], layer.kn[:, :n], layer.vc[:, :n], layer.vn[:, :n],
                           layer.q.c_list, layer.q.rot, scale=scale)
        return out[None, :, None]


class TurboKVCache(DynamicCache):
    """Compressed KV cache for every layer; `bits` per coordinate (TurboQuant MSE quantizer)."""

    def __init__(self, model, capacity, bits=4, seed=0, stage_on_cpu=False, fused=True):
        super().__init__(config=model.config)
        cfg = model.config
        hk, d = cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
        dev, dtype = model.lm_head.weight.device, model.lm_head.weight.dtype
        quant = TurboQuantizer(d, bits, dev, seed)
        self.layers = [TurboLayer(quant, hk, d, capacity, dtype, dev, stage_on_cpu=stage_on_cpu, fused=fused)
                       for _ in model.model.layers]
        self.bits = bits

    def after_prefill(self):
        """Move codes staged on the CPU during prefill to the GPU (no-op if they are already there)."""
        for layer in self.layers:
            layer.kc, layer.vc, layer.kn, layer.vn = (x.to(layer.device) for x in (layer.kc, layer.vc, layer.kn, layer.vn))

    def stored_gb(self, bits=None):
        """Memory the codes + norms would take at `bits` per coordinate (actual storage packs 4 bits)."""
        layer = self.layers[0]
        hk, d = layer.kc.shape[0], layer.q.d
        per_tok = hk * 2 * (d * (bits or self.bits) / 8 + 2)
        return per_tok * layer.n * len(self.layers) / 1e9
