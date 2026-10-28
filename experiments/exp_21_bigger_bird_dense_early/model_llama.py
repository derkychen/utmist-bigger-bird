"""Exp 19 Bigger Bird Flash with full causal attention in a few early layers.

Layer-wise oracle runs (scripts/oracle_exp19_needle.py) showed that exp 19's
multikey NIAH failures come from layers 1-2 missing the needle: forcing the
needle into layers 0-2 recovers most errors, other layers have no effect.
This variant runs those layers dense (SDPA, the same causal operation as exp 0)
and keeps every other layer on exp 19's routing and Triton kernel unchanged.
No answer information is used.
"""
import torch.nn.functional as F

from experiments.exp_19_bigger_bird_flash.model_llama import BiggerBirdFlashAttention


class BiggerBirdDenseEarlyAttention(BiggerBirdFlashAttention):
    def __init__(self, base_attn, *, dense_layers=(0, 1, 2), **kwargs):
        super().__init__(base_attn, **kwargs)
        self.dense_layers = frozenset(int(x) for x in dense_layers)

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads, is_causal=False):
        if self.layer_idx not in self.dense_layers:
            return super().sparse_attention(Q, K, V, token_mask, bsz, num_heads, is_causal)
        if self.training or not is_causal:
            raise ValueError("Dense early layers support causal inference only")
        if token_mask is not None and not bool(token_mask.all()):
            raise ValueError("Dense early layers require an unpadded single prompt")
        q_len, dim = Q.shape[1:]
        n = K.shape[1]
        if q_len not in (n, 1):
            raise ValueError("Dense early layers support full prefill or one-token decode")
        q = Q.view(bsz, num_heads, q_len, dim)
        k = K.view(bsz, num_heads, n, dim)
        v = V.view(bsz, num_heads, n, dim)
        # Q is already multiplied by the model's attention scale, so scale=1.
        # A single decode query may attend to every cached key.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=q_len == n, scale=1.0)
        self.last_selected_indices = None
        self.last_diagnostics = dict(backend="sdpa_dense", dense_layer=True,
                                     query_tokens=q_len, source_tokens=n)
        return out.reshape(bsz * num_heads, q_len, dim)
