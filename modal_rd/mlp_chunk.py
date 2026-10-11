"""Sequence-chunked MLP for long-context prefill.

LlamaMLP at ~127K tokens materialises gate/up/act tensors of [1, n, 14336] (~3.7 GB each in BF16)
even though the attention path is already O(n) in memory. Running the MLP in slices of `chunk`
tokens is exact (the MLP is position-wise) and bounds those transients. It helps dense and sparse
arms equally, so report it for both when comparing memory.
"""
import torch


def install_chunked_mlp(model, chunk=8192):
    """Wrap every decoder layer's MLP; idempotent. chunk <= 0 restores the original forward."""
    for layer in model.model.layers:
        mlp = layer.mlp
        base = getattr(mlp, "_unchunked_forward", mlp.forward)
        mlp._unchunked_forward = base
        if chunk <= 0:
            mlp.forward = base
            continue

        def forward(x, _base=base):
            n = x.shape[-2]
            if n <= chunk:
                return _base(x)
            out = torch.empty_like(x)
            for s in range(0, n, chunk):
                out[..., s:s + chunk, :] = _base(x[..., s:s + chunk, :])
            return out

        mlp.forward = forward
    return model
