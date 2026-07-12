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


def _effective_num_blocks(num_blocks: int, seq_len: int, block_size: int) -> int:
    n_blocks_k = max(1, seq_len // block_size)
    return min(num_blocks, max(2, n_blocks_k // 2))


class DeepSeekPBSAttention(BartAttention):
    """Low-rank block routing (head-shared) + PBS sorted token indices + sparse softmax."""

    def __init__(
        self,
        base_attn: BartAttention,
        top_k: int = 64,
        low_rank_dim: int = 16,
        block_size: int = 32,
        num_blocks: int = 4,
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
        self.block_size = block_size
        self.num_blocks = num_blocks
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
        src_len = K.size(1)
        k_eff = effective_top_k(self.top_k, src_len)
        M_blocks = _effective_num_blocks(self.num_blocks, src_len, self.block_size)
        token_mask = token_mask_1d(attention_mask, bsz, src_len, Q.device)

        if src_len <= k_eff or src_len < self.block_size * 2:
            out = dense_self_attention(
                Q, K, V, attention_mask, bsz, self.num_heads, self.dropout, self.training
            )
        else:
            n_blocks_k = max(1, src_len // self.block_size)
            usable_src = n_blocks_k * self.block_size
            d_low = min(self.low_rank_dim, self.head_dim)
            K_low = K[:, :usable_src, :d_low]
            K_blocks = K_low.view(BH, n_blocks_k, self.block_size, d_low).mean(dim=2)
            q_mean = Q[:, :, :d_low].mean(dim=1, keepdim=True)
            block_scores = torch.bmm(q_mean, K_blocks.transpose(1, 2)).squeeze(1) / (d_low ** 0.5)
            if token_mask is not None:
                block_mask = token_mask[:, :usable_src].view(bsz, n_blocks_k, self.block_size).any(dim=-1)
                block_mask = block_mask.unsqueeze(1).expand(bsz, self.num_heads, n_blocks_k).reshape(BH, n_blocks_k)
                block_scores = block_scores.masked_fill(~block_mask, -1e9)
            M = min(M_blocks, n_blocks_k)
            _, top_blocks = torch.topk(block_scores, k=M, dim=-1)
            offs = torch.arange(self.block_size, device=Q.device)
            base = top_blocks.unsqueeze(-1) * self.block_size
            top_idx = (base.unsqueeze(-1) + offs.view(1, 1, -1)).reshape(BH, M * self.block_size)
            top_idx, _ = torch.sort(top_idx, dim=-1)
            if top_idx.size(-1) > k_eff:
                d_low_full = min(self.low_rank_dim, self.head_dim)
                bh = torch.arange(BH, device=Q.device).view(BH, 1)
                K_sub = K[bh, top_idx, :d_low_full]
                rough_tok = torch.bmm(
                    Q[:, :, :d_low_full].mean(dim=1, keepdim=True),
                    K_sub.transpose(1, 2),
                ).squeeze(1)
                _, pick = torch.topk(rough_tok, k=k_eff, dim=-1)
                top_idx = torch.gather(top_idx, 1, pick)
                top_idx, _ = torch.sort(top_idx, dim=-1)
            out = sdpa_head_shared_or_none(
                Q, K, V, top_idx, attention_mask, bsz, self.num_heads,
                self.use_triton, self.training,
            )
            if out is None:
                out = sparse_attention_head_shared(
                    Q, K, V, top_idx, self.dropout, self.training, token_mask, bsz, self.num_heads
                )

        attn_output = out.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(
            bsz, tgt_len, self.embed_dim
        )
        attn_output = self.out_proj(attn_output)
        return (attn_output, None)


def patch_bart(model: nn.Module, **kw):
    def _rec(m):
        for n, c in list(m.named_children()):
            if isinstance(c, BartAttention):
                if getattr(c, "is_decoder", False):
                    continue
                setattr(m, n, DeepSeekPBSAttention(c, **kw))
            else:
                _rec(c)

    _rec(model)


class PatchedModel(nn.Module):
    def __init__(self, base_model, top_k=64, low_rank_dim=16, block_size=32, num_blocks=4, use_triton=True):
        super().__init__()
        self.model = base_model
        self.use_triton = use_triton
        patch_bart(
            self.model,
            top_k=top_k,
            low_rank_dim=low_rank_dim,
            block_size=block_size,
            num_blocks=num_blocks,
            use_triton=use_triton,
        )

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
