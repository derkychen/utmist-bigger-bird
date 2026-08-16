"""Exp 2 — Lightning Hybrid attention on R1-Distill-Llama-8B.

Combines a local sliding-window branch (block_size=128) with a global ELU
linear-attention branch for long-range context.  For sequences shorter than
``block_size * 4`` only the local branch is used; otherwise the output is
``local + 0.5 * global``.

This is the Llama-3 port of the original BART-based experiment.

Key changes from the BART version:
  - Inherits from ``LlamaSparseAttention`` (handles GQA, RoPE, projections)
  - ``sparse_attention()`` receives already-projected, RoPE'd, GQA-expanded
    Q/K/V as [BH, T, d] — same interface the sparse_attn_utils expect
  - Bidirectional attention (no causal mask) for sequence classification
  - LoRA training instead of full fine-tuning
  - use_triton=False (PyTorch fallback; safer on MIG slices)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from patches.llama.llama_patched_model import (
    LlamaSparseAttention,
    patch_llama,
    LlamaPatchedModel,
    apply_lora,
)
from sparse_attn_utils import (
    causal_sparse_attention,
    last_query_topk_indices,
)


class LightningHybridAttention(LlamaSparseAttention):
    def __init__(self, base_attn, block_size: int = 128, use_triton: bool = False):
        super().__init__(base_attn)
        self.block_size = block_size
        self.use_triton = use_triton

    # ------------------------------------------------------------------
    # Local branch — sliding-window softmax attention via unfold
    # ------------------------------------------------------------------
    def _windowed_softmax_attention(self, Q, K, V, token_mask, bsz, tgt_len, src_len):
        """Sliding-window attention via unfold: O(T * W * d), no T x T tensor.

        Mirrors the fused ``band_attention`` kernel: query t attends keys in the
        symmetric band [t-radius, t+radius] intersected with [0, src_len-1]
        (out-of-range boundary positions are masked, not treated as zero sinks).
        """
        half = min(self.block_size // 2, src_len - 1) if src_len > 0 else 0
        w = 2 * half + 1
        device, dtype = Q.device, Q.dtype
        BH = Q.size(0)
        neg_inf = torch.finfo(dtype).min

        K_pad = F.pad(K, (0, 0, half, half))
        V_pad = F.pad(V, (0, 0, half, half))
        K_win = K_pad.unfold(1, w, 1)[:, :tgt_len].transpose(-1, -2)
        V_win = V_pad.unfold(1, w, 1)[:, :tgt_len].transpose(-1, -2)

        # Q is already pre-scaled by self.scaling in the base class.
        scores = torch.einsum("btd,btwd->btw", Q, K_win)

        q_pos = torch.arange(tgt_len, device=device).view(1, -1, 1)
        col_off = torch.arange(w, device=device).view(1, 1, -1) - half
        key_pos = q_pos + col_off
        in_range = (key_pos >= 0) & (key_pos <= src_len - 1)
        valid = in_range.expand(BH, -1, -1)

        if token_mask is not None:
            key_pos_c = key_pos.clamp(0, src_len - 1)
            am = token_mask.unsqueeze(1).unsqueeze(1).expand(
                bsz, self.num_heads, tgt_len, src_len
            )
            am = am.reshape(BH, tgt_len, src_len)
            key_allowed = torch.gather(am, 2, key_pos_c.expand(BH, -1, -1))
            valid = valid & key_allowed

        scores = scores.masked_fill(~valid, neg_inf)
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=0.0, training=self.training)
        return torch.einsum("btw,btwd->btd", attn, V_win)

    # ------------------------------------------------------------------
    # Global branch — ELU-feature linear attention
    # ------------------------------------------------------------------
    def _linear_global_attention(self, Q, K, V, token_mask, bsz, src_len):
        """ELU-feature linear attention (global long-range branch): O(S * d^2)."""
        BH = Q.size(0)
        Q_l = F.elu(Q) + 1.0
        K_l = F.elu(K) + 1.0
        if token_mask is not None:
            pm = token_mask.unsqueeze(1).expand(
                bsz, self.num_heads, src_len
            ).reshape(BH, src_len)
            K_l = K_l * pm.unsqueeze(-1)
        KV = torch.bmm(K_l.transpose(1, 2), V)
        Z = K_l.sum(dim=1, keepdim=True)
        Num = torch.bmm(Q_l, KV)
        Den = torch.bmm(Q_l, Z.transpose(1, 2))
        return Num / (Den + 1e-6)

    # ------------------------------------------------------------------
    # Causal local branch — sliding-window with causal masking
    # ------------------------------------------------------------------
    def _causal_windowed_softmax_attention(self, Q, K, V, token_mask, bsz, tgt_len, src_len):
        """Causal sliding-window attention: query t attends to keys [max(0, t-W+1), t]."""
        W = min(self.block_size, src_len)
        device, dtype = Q.device, Q.dtype
        BH = Q.size(0)
        neg_inf = torch.finfo(dtype).min

        # Build causal local window indices: [T, W]
        positions = torch.arange(tgt_len, device=device)
        local_start = (positions - W + 1).clamp(min=0)
        local_offsets = torch.arange(W, device=device).unsqueeze(0)  # [1, W]
        local_idx = (local_start.unsqueeze(1) + local_offsets).clamp(max=src_len - 1)  # [T, W]

        # Gather K, V: [BH, T, W, d]
        local_idx_exp = local_idx.unsqueeze(0).expand(BH, -1, -1)  # [BH, T, W]
        idx_gather = local_idx_exp.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        K_win = torch.gather(
            K.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx_gather
        )
        V_win = torch.gather(
            V.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx_gather
        )

        # Scores: [BH, T, W]
        scores = torch.einsum("btd,btwd->btw", Q, K_win)

        # Causal mask: key position must be <= query position
        q_pos = positions.view(1, -1, 1)  # [1, T, 1]
        causal_allowed = local_idx.unsqueeze(0) <= q_pos  # [BH, T, W]
        scores = scores.masked_fill(~causal_allowed, neg_inf)

        # Padding mask
        if token_mask is not None:
            am = token_mask.unsqueeze(1).expand(bsz, self.num_heads, src_len).reshape(BH, src_len)
            allowed_pad = torch.gather(
                am.unsqueeze(1).expand(-1, tgt_len, -1), 2, local_idx_exp
            )
            scores = scores.masked_fill(~allowed_pad, neg_inf)

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=0.0, training=self.training)
        return torch.einsum("btw,btwd->btd", attn, V_win)

    # ------------------------------------------------------------------
    # sparse_attention — called by the base class
    # ------------------------------------------------------------------
    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads, is_causal=False):
        BH, tgt_len, _ = Q.shape
        src_len = K.size(1)

        # Causal mode: causal local window + top-k routed global keys
        # (linear attention fails for precise retrieval — use content-based routing instead)
        if is_causal:
            k_global = min(256, src_len)  # global top-k keys
            d_low = min(self.low_rank_dim if hasattr(self, 'low_rank_dim') else 64, self.head_dim)
            Q_low = Q[:, :, :d_low]
            K_low = K[:, :, :d_low]
            routed_idx = last_query_topk_indices(
                Q_low, K_low, k_global, token_mask, bsz, num_heads,
            )
            return causal_sparse_attention(
                Q, K, V, routed_idx, local_window=self.block_size,
                token_mask=token_mask, bsz=bsz, num_heads=num_heads,
            )

        # Bidirectional mode (original)
        local_out = self._windowed_softmax_attention(
            Q, K, V, token_mask, bsz, tgt_len, src_len
        )

        if tgt_len <= self.block_size * 4:
            out = local_out
        else:
            # Global long-range branch -> ELU linear attention
            global_out = self._linear_global_attention(
                Q, K, V, token_mask, bsz, src_len
            )
            out = local_out + 0.5 * global_out

        return out


def build_model(
    model_path: str = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B"),
    block_size: int = 128,
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """Build the patched R1-8B model with Lightning Hybrid attention + LoRA."""
    model = LlamaPatchedModel.from_pretrained(
        model_path=model_path,
        attention_cls=LightningHybridAttention,
        num_labels=num_labels,
        attn_kwargs={
            "block_size": block_size,
            "use_triton": False,  # safer on MIG; PyTorch fallback works
        },
    )
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha)
    return model
