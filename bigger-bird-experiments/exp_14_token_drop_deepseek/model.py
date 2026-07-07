import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput
from transformers.models.bart.modeling_bart import BartAttention

from shared.sparse_attn_utils import dense_self_attention, sdpa_dense_or_none
from exp_1_deepseek_topk.model import DeepSeekTopKAttention

# Idea: Token Dropping + DeepSeek Top-K Sparse Attention
# After early dense layers extract local syntax, drop low-importance tokens.
# Then use DeepSeek-style low-rank top-K routing for the remaining layers.
# This gives TWO sparsity wins: (1) shorter sequence, (2) fewer keys per query.


class DenseKernelAttention(BartAttention):
    """Standard dense attention via F.scaled_dot_product_attention."""

    def __init__(self, base_attn: BartAttention, use_triton: bool = True):
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
        for k in ("cache_position", "position_bias", "alibi_bias"):
            kwargs.pop(k, None)

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

        out = sdpa_dense_or_none(
            Q, K, V, attention_mask, bsz, self.num_heads, self.use_triton, self.training
        )
        if out is None:
            out = dense_self_attention(
                Q, K, V, attention_mask, bsz, self.num_heads, self.dropout, self.training
            )

        attn_output = out.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(
            bsz, tgt_len, self.embed_dim
        )
        return (self.out_proj(attn_output), None)


def _patch_layer_attn(layer, new_attn):
    """Replace self_attn on a single encoder layer."""
    layer.self_attn = new_attn


class TokenDropSparseEncoder(nn.Module):
    """Wraps a BartEncoder: early layers dense, drop tokens, late layers DeepSeek top-k."""

    def __init__(
        self,
        encoder: nn.Module,
        drop_after_layer: int = 3,
        drop_ratio: float = 0.3,
        top_k: int = 64,
        low_rank_dim: int = 16,
        use_triton: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.drop_after_layer = drop_after_layer
        self.drop_ratio = drop_ratio
        self.top_k = top_k
        self.low_rank_dim = low_rank_dim
        self.use_triton = use_triton

        # Patch early layers with dense attention, late layers with DeepSeek top-k
        for i, layer in enumerate(encoder.layers):
            sa = getattr(layer, "self_attn", None)
            if not isinstance(sa, BartAttention) or getattr(sa, "is_decoder", False):
                continue
            if i < drop_after_layer:
                _patch_layer_attn(layer, DenseKernelAttention(sa, use_triton=use_triton))
            else:
                _patch_layer_attn(
                    layer,
                    DeepSeekTopKAttention(
                        sa, top_k=top_k, low_rank_dim=low_rank_dim, use_triton=use_triton
                    ),
                )

    def forward(self, input_ids=None, attention_mask=None, return_dict=True, **kwargs):
        bsz, seq_len = input_ids.shape

        inputs_embeds = self.encoder.embed_tokens(input_ids)
        embed_pos = self.encoder.embed_positions(input_ids)
        embed_pos = embed_pos.to(inputs_embeds.device)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.encoder.layernorm_embedding(hidden_states)
        dropout_p = self.encoder.config.dropout
        hidden_states = F.dropout(hidden_states, p=dropout_p, training=self.training)

        if attention_mask is None:
            current_mask = torch.ones(bsz, seq_len, device=input_ids.device, dtype=torch.long)
        else:
            current_mask = attention_mask
        kept_indices = None

        for i, layer in enumerate(self.encoder.layers):
            cur_len = hidden_states.size(1)
            mask_f = current_mask.float() if current_mask.dtype != torch.float else current_mask
            ext_mask = (1.0 - mask_f) * torch.finfo(hidden_states.dtype).min
            ext_mask = ext_mask[:, None, None, :]  # [B, 1, 1, cur_len]
            out = layer(hidden_states, attention_mask=ext_mask, layer_head_mask=None, output_attentions=False)
            hidden_states = out[0] if isinstance(out, tuple) else out

            # Drop tokens AFTER this layer if it's the drop point
            if i + 1 == self.drop_after_layer and self.drop_ratio > 0:
                norms = hidden_states.norm(dim=-1)  # [B, T]
                pad_mask = (current_mask == 0)
                norms = norms.masked_fill(pad_mask, -1e9)
                cur_len = hidden_states.size(1)
                keep_n = max(1, int(cur_len * (1.0 - self.drop_ratio)))
                _, top_idx = torch.topk(norms, k=keep_n, dim=-1)  # [B, keep_n]
                top_idx, _ = torch.sort(top_idx, dim=-1)  # preserve relative order
                gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
                hidden_states = torch.gather(hidden_states, 1, gather_idx)
                current_mask = torch.gather(current_mask, 1, top_idx)
                kept_indices = top_idx

        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=None, attentions=None), current_mask, kept_indices


class AttnPool(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x, mask):
        h = torch.tanh(self.proj(x))
        s = self.score(h).squeeze(-1)
        s = s.masked_fill(~mask.bool(), torch.finfo(s.dtype).min)
        a = torch.softmax(s, dim=-1)
        return torch.bmm(a.unsqueeze(1), x).squeeze(1)


class PatchedModel(nn.Module):
    def __init__(
        self,
        base_model,
        drop_after_layer: int = 3,
        drop_ratio: float = 0.3,
        top_k: int = 64,
        low_rank_dim: int = 16,
        use_triton: bool = True,
    ):
        super().__init__()
        self.model = base_model
        self.use_triton = use_triton
        encoder = base_model.model.encoder
        self.sparse_encoder = TokenDropSparseEncoder(
            encoder,
            drop_after_layer=drop_after_layer,
            drop_ratio=drop_ratio,
            top_k=top_k,
            low_rank_dim=low_rank_dim,
            use_triton=use_triton,
        )
        hidden = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "d_model")
        self.attn_pool = AttnPool(hidden)
        self.drop_after_layer = drop_after_layer
        self.drop_ratio = drop_ratio
        self.top_k = top_k
        self.low_rank_dim = low_rank_dim

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
        encoder_out, final_mask, _ = self.sparse_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last = encoder_out.last_hidden_state
        if final_mask is None:
            final_mask = torch.ones(last.size()[:2], device=last.device, dtype=torch.long)
        pooled = self.attn_pool(last, final_mask)
        logits = self.model.classification_head(pooled)
        loss = None
        if labels is not None:
            if labels.dtype != torch.long:
                labels = labels.long()
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits)

    @property
    def config(self):
        return self.model.config
