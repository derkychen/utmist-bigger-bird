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


class DeepSeekTopKAttention(BartAttention):
    def __init__(self, base_attn: BartAttention, top_k: int = 128, low_rank_dim: int = 16, use_triton: bool = True):
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
        self.use_triton = use_triton

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        _ = kwargs.pop("cache_position", None)
        _ = kwargs.pop("position_bias", None)
        _ = kwargs.pop("alibi_bias", None)

        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * (self.head_dim ** -0.5)
        if is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        BH = bsz * self.num_heads
        Q = self._shape(query_states, tgt_len, bsz).reshape(BH, tgt_len, self.head_dim)
        K = key_states.reshape(BH, -1, self.head_dim)
        V = value_states.reshape(BH, -1, self.head_dim)
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
        attn_output = self.out_proj(attn_output)
        return (attn_output, None)


def patch_bart(model: nn.Module, top_k: int = 128, low_rank_dim: int = 16, use_triton: bool = True):
    def _recurse(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, BartAttention):
                if getattr(child, "is_decoder", False):
                    continue
                setattr(module, name, DeepSeekTopKAttention(child, top_k, low_rank_dim, use_triton=use_triton))
            else:
                _recurse(child)

    _recurse(model)


class PatchedModel(nn.Module):
    def __init__(self, base_model, top_k=128, low_rank_dim=16, use_triton=True):
        super().__init__()
        self.model = base_model
        self.use_triton = use_triton
        patch_bart(self.model, top_k=top_k, low_rank_dim=low_rank_dim, use_triton=use_triton)

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
