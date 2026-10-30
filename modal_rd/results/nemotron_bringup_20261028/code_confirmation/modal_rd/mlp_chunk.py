"""Sequence-chunked MLP and RMSNorm for long-context prefill.

LlamaMLP at ~127K tokens materialises gate/up/act tensors of [1, n, 14336] (~3.7 GB each in BF16)
even though the attention path is already O(n) in memory. LlamaRMSNorm upcasts its whole input to
FP32 and keeps two [1, n, 4096] FP32 temporaries (~1.1 GB each at 65K tokens). Both ops are
position-wise, so running them in slices of `chunk` tokens is exact and bounds those transients.
It helps dense and sparse arms equally, so report it for both when comparing memory.
"""
import torch


def _chunk(module, chunk):
    base = getattr(module, "_unchunked_forward", module.forward)
    module._unchunked_forward = base
    if chunk <= 0:
        module.forward = base
        return

    def forward(x, _base=base):
        n = x.shape[-2]
        if n <= chunk:
            return _base(x)
        out = torch.empty_like(x)
        for s in range(0, n, chunk):
            out[..., s:s + chunk, :] = _base(x[..., s:s + chunk, :])
        return out

    module.forward = forward


def install_chunked_mlp(model, chunk=8192, norms=True):
    """Wrap every decoder layer's MLP (and, with `norms`, its RMSNorms and the final norm); idempotent.
    chunk <= 0 restores the original forwards."""
    for layer in model.model.layers:
        _chunk(layer.mlp, chunk)
        if norms:
            _chunk(layer.input_layernorm, chunk)
            _chunk(layer.post_attention_layernorm, chunk)
    if norms:
        _chunk(model.model.norm, chunk)
    return model
