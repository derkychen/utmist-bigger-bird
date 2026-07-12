import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import BartAttention

from shared.patched_model import classification_forward
from shared.sparse_attn_utils import (
    dense_self_attention,
    effective_top_k,
    head_shared_topk_indices,
    sdpa_head_shared_or_none,
    sparse_attention_head_shared,
    token_mask_1d,
)


class LayerAdaptiveAttention(BartAttention):
    def __init__(
        self,
        base_attn: BartAttention,
        top_k: int = 64,
        low_rank_dim: int = 16,
        layer_idx: int = -1,
        use_triton: bool = True,
    ):
        super().__init__(
            embed_dim=base_attn.embed_dim,
            num_heads=base_attn.num_heads,
            dropout=base_attn.dropout.p if isinstance(base_attn.dropout, nn.Dropout) else float(base_attn.dropout),
            is_decoder=base_attn.is_decoder,
            bias=base_attn.k_proj.bias is not None,
        )
        self.q_proj.load_state_dict(base_attn.q_proj.state_dict())
        self.k_proj.load_state_dict(base_attn.k_proj.state_dict())
        self.v_proj.load_state_dict(base_attn.v_proj.state_dict())
        self.out_proj.load_state_dict(base_attn.out_proj.state_dict())
        self.top_k = top_k
        self.low_rank_dim = low_rank_dim
        self.layer_idx = layer_idx
        self.use_triton = use_triton

    def _shape(self, tensor, seq_len, bsz):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        for k in ("cache_position", "position_bias", "alibi_bias"):
            kwargs.pop(k, None)

        is_cross = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        Qs = self.q_proj(hidden_states) * (self.head_dim ** -0.5)
        if is_cross:
            Ks = self._shape(self.k_proj(key_value_states), -1, bsz)
            Vs = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            Ks = self._shape(self.k_proj(hidden_states), -1, bsz)
            Vs = self._shape(self.v_proj(hidden_states), -1, bsz)

        BH = bsz * self.num_heads
        Q = self._shape(Qs, tgt_len, bsz).reshape(BH, tgt_len, self.head_dim)
        K = Ks.reshape(BH, -1, self.head_dim)
        V = Vs.reshape(BH, -1, self.head_dim)
        src_len = K.size(1)
        k_eff = effective_top_k(self.top_k, src_len)
        token_mask = token_mask_1d(attention_mask, bsz, src_len, Q.device)

        if src_len <= k_eff:
            out = dense_self_attention(
                Q, K, V, attention_mask, bsz, self.num_heads, self.dropout, self.training
            )
        else:
            d_low = min(self.low_rank_dim, self.head_dim)
            Q_low = Q[:, :, :d_low]
            K_low = K[:, :, :d_low]
            topk_idx = head_shared_topk_indices(
                Q_low, K_low, k_eff, token_mask, bsz, self.num_heads
            )
            out = sdpa_head_shared_or_none(
                Q, K, V, topk_idx, attention_mask, bsz, self.num_heads,
                self.use_triton, self.training,
            )
            if out is None:
                out = sparse_attention_head_shared(
                    Q, K, V, topk_idx, self.dropout, self.training, token_mask, bsz, self.num_heads
                )

        attn_output = out.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(
            bsz, tgt_len, self.embed_dim
        )
        return (self.out_proj(attn_output), None)


def _schedule(layer_idx: int, n_layers: int, k_early: int, k_mid: int, k_late: int):
    third = n_layers / 3
    if layer_idx < third:
        return k_early
    if layer_idx < 2 * third:
        return k_mid
    return k_late


def patch_bart(model: nn.Module, k_early=192, k_mid=64, k_late=32, low_rank_dim=16, n_layers_hint=12, use_triton=True):
    encoder = getattr(model, "model", model)
    if hasattr(encoder, "encoder"):
        encoder = encoder.encoder
    layers = getattr(encoder, "layers", None)
    if layers is None:
        n = n_layers_hint
        idx_holder = {"i": 0}

        def _rec(m):
            for nm, c in list(m.named_children()):
                if isinstance(c, BartAttention) and not getattr(c, "is_decoder", False):
                    k = _schedule(idx_holder["i"], n, k_early, k_mid, k_late)
                    setattr(
                        m,
                        nm,
                        LayerAdaptiveAttention(c, top_k=k, low_rank_dim=low_rank_dim, layer_idx=idx_holder["i"], use_triton=use_triton),
                    )
                    idx_holder["i"] += 1
                else:
                    _rec(c)

        _rec(model)
        return idx_holder["i"]

    n = len(layers)
    for i, layer in enumerate(layers):
        k = _schedule(i, n, k_early, k_mid, k_late)
        layer.self_attn = LayerAdaptiveAttention(
            layer.self_attn, top_k=k, low_rank_dim=low_rank_dim, layer_idx=i, use_triton=use_triton
        )
    return n


class PatchedModel(nn.Module):
    def __init__(self, base_model, k_early=192, k_mid=64, k_late=32, low_rank_dim=16, use_triton=True):
        super().__init__()
        self.model = base_model
        self.use_triton = use_triton
        patch_bart(self.model, k_early=k_early, k_mid=k_mid, k_late=k_late, low_rank_dim=low_rank_dim, use_triton=use_triton)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            return self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            return self.model.gradient_checkpointing_disable()

    @property
    def supports_gradient_checkpointing(self):
        return getattr(self.model, "supports_gradient_checkpointing", True)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return classification_forward(self.model, input_ids, attention_mask, labels, **kwargs)

    @property
    def config(self):
        return self.model.config
