import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import BartAttention

from shared.kernels import should_use_train_kernel, should_use_triton
from shared.patched_model import classification_forward
from shared.sparse_attn_utils import (
    dense_self_attention,
    gather_attention_triton_or_none,
    token_mask_1d,
)


class DynamicGlobalAttention(BartAttention):
    def __init__(self, base_attn: BartAttention, window_size: int = 64, num_globals: int = 16, use_triton: bool = True):
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
        self.window_size = window_size
        self.num_globals = num_globals
        self.use_triton = use_triton
        self.global_gate = nn.Linear(self.embed_dim, 1)

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
        M = self.window_size + self.num_globals
        token_mask = token_mask_1d(attention_mask, bsz, src_len, Q.device)

        if src_len <= M:
            out = dense_self_attention(
                Q, K, V, attention_mask, bsz, self.num_heads, self.dropout, self.training
            )
        else:
            global_scores = self.global_gate(hidden_states).squeeze(-1)
            if token_mask is not None:
                global_scores = global_scores.masked_fill(~token_mask, -1e9)
            g = min(self.num_globals, src_len)
            _, global_idx = torch.topk(global_scores, k=g, dim=-1)
            global_bh = global_idx.unsqueeze(1).expand(bsz, self.num_heads, g).reshape(BH, g)

            half = self.window_size // 2
            w = min(self.window_size, src_len)

            # Inference fast path: merge globals + window into one gathered key set.
            # exp_3's sparse branch divides scores by sqrt(d) on top of the pre-scaled
            # Q, so pass scale=head_dim**-0.5 to stay equivalent to the PyTorch path.
            if should_use_triton(self.use_triton, Q, training=self.training) or (
                self.training and should_use_train_kernel(self.use_triton, Q)
            ):
                q_pos_k = torch.arange(tgt_len, device=Q.device).view(1, -1, 1)
                col_off_k = torch.arange(w, device=Q.device).view(1, 1, -1) - half
                win_pos_k = (q_pos_k + col_off_k).clamp(0, src_len - 1).expand(BH, -1, -1)
                g_pos_k = global_bh.unsqueeze(1).expand(-1, tgt_len, -1)
                token_idx = torch.cat([g_pos_k, win_pos_k], dim=-1)
                out = gather_attention_triton_or_none(
                    Q, K, V, token_idx, attention_mask, bsz, self.num_heads,
                    self.use_triton, self.training, scale=self.head_dim ** -0.5,
                )
                if out is not None:
                    attn_output = out.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(
                        bsz, tgt_len, self.embed_dim
                    )
                    return (self.out_proj(attn_output), None)

            bh = torch.arange(BH, device=Q.device).view(BH, 1)
            K_g = K[bh, global_bh, :]
            V_g = V[bh, global_bh, :]

            K_pad = F.pad(K, (0, 0, half, half))
            V_pad = F.pad(V, (0, 0, half, half))
            K_win = K_pad.unfold(1, w, 1)[:, :tgt_len].transpose(-1, -2)
            V_win = V_pad.unfold(1, w, 1)[:, :tgt_len].transpose(-1, -2)

            scores_g = torch.einsum("btd,bgd->btg", Q, K_g) / (self.head_dim ** 0.5)
            scores_w = torch.einsum("btd,btwd->btw", Q, K_win) / (self.head_dim ** 0.5)
            scores = torch.cat([scores_g, scores_w], dim=-1)

            if token_mask is not None:
                g_pos = global_bh.unsqueeze(1).expand(-1, tgt_len, -1)
                q_pos = torch.arange(tgt_len, device=Q.device).view(1, -1, 1)
                col_off = torch.arange(w, device=Q.device).view(1, 1, -1) - half
                win_pos = (q_pos + col_off).clamp(0, src_len - 1)
                key_pos = torch.cat([g_pos, win_pos.expand(BH, -1, -1)], dim=-1)
                am = token_mask.unsqueeze(1).unsqueeze(1).expand(bsz, self.num_heads, tgt_len, src_len)
                am = am.reshape(BH, tgt_len, src_len)
                allowed = torch.gather(am, 2, key_pos)
                scores = scores.masked_fill(~allowed, -1e9)

            attn = F.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            attn_g, attn_w = attn.split([g, w], dim=-1)
            out = torch.einsum("btg,bgd->btd", attn_g, V_g) + torch.einsum("btw,btwd->btd", attn_w, V_win)

        attn_output = out.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(
            bsz, tgt_len, self.embed_dim
        )
        attn_output = self.out_proj(attn_output)
        return (attn_output, None)


def patch_bart(model: nn.Module, window_size: int = 64, num_globals: int = 16, use_triton: bool = True):
    def _recurse(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, BartAttention):
                if getattr(child, "is_decoder", False):
                    continue
                setattr(module, name, DynamicGlobalAttention(child, window_size, num_globals, use_triton=use_triton))
            else:
                _recurse(child)

    _recurse(model)


class PatchedModel(nn.Module):
    def __init__(self, base_model, window_size=64, num_globals=16, use_triton=True):
        super().__init__()
        self.model = base_model
        self.use_triton = use_triton
        patch_bart(self.model, window_size=window_size, num_globals=num_globals, use_triton=use_triton)

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
