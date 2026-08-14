"""Exp 3 — Dynamic Global Attention on R1-Distill-Llama-8B.

Local window attention (window_size=64) combined with learned global tokens
(num_globals=16) selected by a gating network. This is the Llama-3 port of the
original BART-based experiment.

Key changes from the BART version:
  - Inherits from ``LlamaSparseAttention`` (handles GQA, RoPE, projections)
  - ``sparse_attention()`` receives already-projected, RoPE'd, GQA-expanded
    Q/K/V as [BH, T, d] — Q is pre-scaled by self.scaling
  - Bidirectional attention (no causal mask) for sequence classification
  - LoRA training instead of full fine-tuning
  - The global gate uses ``self.config.hidden_size`` (4096 for Llama-8B) instead
    of ``self.embed_dim`` (768 for BART-base)
  - ``forward()`` is overridden to capture hidden_states for the gate
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from patches.llama.llama_patched_model import (
    LlamaSparseAttention,
    LlamaPatchedModel,
    apply_lora,
)
from sparse_attn_utils import (
    dense_self_attention,
)


class DynamicGlobalAttention(LlamaSparseAttention):
    def __init__(
        self,
        base_attn,
        window_size: int = 64,
        num_globals: int = 16,
        use_triton: bool = True,
    ):
        super().__init__(base_attn)
        self.window_size = window_size
        self.num_globals = num_globals
        self.use_triton = use_triton
        self.global_gate = nn.Linear(self.config.hidden_size, 1)
        self._hidden_states = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        # Capture hidden_states so sparse_attention can compute the global gate
        self._hidden_states = hidden_states
        return super().forward(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads, is_causal=False):
        if is_causal:
            return dense_self_attention(Q, K, V, token_mask, bsz, num_heads, 0.0, self.training, is_causal=True)
        BH, tgt_len, _ = Q.shape
        src_len = K.size(1)
        M = self.window_size + self.num_globals

        if src_len <= M:
            # Fall back to dense attention on short sequences
            return dense_self_attention(
                Q, K, V, None, bsz, num_heads, 0.0, self.training
            )

        # --- Select global tokens via learned gate (head-shared, per-batch) ---
        hidden_states = self._hidden_states  # [B, T, hidden_size]
        global_scores = self.global_gate(hidden_states).squeeze(-1)  # [B, T]
        if token_mask is not None:
            global_scores = global_scores.masked_fill(~token_mask, -1e9)
        g = min(self.num_globals, src_len)
        _, global_idx = torch.topk(global_scores, k=g, dim=-1)  # [B, g]
        global_bh = global_idx.unsqueeze(1).expand(bsz, num_heads, g).reshape(BH, g)

        # --- Local window ---
        half = self.window_size // 2
        w = min(self.window_size, src_len)

        bh = torch.arange(BH, device=Q.device).view(BH, 1)
        K_g = K[bh, global_bh, :]  # [BH, g, d]
        V_g = V[bh, global_bh, :]

        K_pad = F.pad(K, (0, 0, half, half))
        V_pad = F.pad(V, (0, 0, half, half))
        K_win = K_pad.unfold(1, w, 1)[:, :tgt_len].transpose(-1, -2)  # [BH, T, w, d]
        V_win = V_pad.unfold(1, w, 1)[:, :tgt_len].transpose(-1, -2)

        # Q is pre-scaled; exp_3 additionally divides scores by sqrt(d)
        scores_g = torch.einsum("btd,bgd->btg", Q, K_g) / (self.head_dim ** 0.5)
        scores_w = torch.einsum("btd,btwd->btw", Q, K_win) / (self.head_dim ** 0.5)
        scores = torch.cat([scores_g, scores_w], dim=-1)

        # --- Apply padding mask ---
        if token_mask is not None:
            g_pos = global_bh.unsqueeze(1).expand(-1, tgt_len, -1)
            q_pos = torch.arange(tgt_len, device=Q.device).view(1, -1, 1)
            col_off = torch.arange(w, device=Q.device).view(1, 1, -1) - half
            win_pos = (q_pos + col_off).clamp(0, src_len - 1)
            key_pos = torch.cat([g_pos, win_pos.expand(BH, -1, -1)], dim=-1)
            am = token_mask.unsqueeze(1).unsqueeze(1).expand(
                bsz, num_heads, tgt_len, src_len
            )
            am = am.reshape(BH, tgt_len, src_len)
            allowed = torch.gather(am, 2, key_pos)
            scores = scores.masked_fill(~allowed, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=0.0, training=self.training)
        attn_g, attn_w = attn.split([g, w], dim=-1)
        out = (
            torch.einsum("btg,bgd->btd", attn_g, V_g)
            + torch.einsum("btw,btwd->btd", attn_w, V_win)
        )
        return out


def build_model(
    model_path: str = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B"),
    window_size: int = 64,
    num_globals: int = 16,
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """Build the patched R1-8B model with Dynamic Global attention + LoRA."""
    model = LlamaPatchedModel.from_pretrained(
        model_path=model_path,
        attention_cls=DynamicGlobalAttention,
        num_labels=num_labels,
        attn_kwargs={
            "window_size": window_size,
            "num_globals": num_globals,
            "use_triton": False,
        },
    )
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha)
    return model
