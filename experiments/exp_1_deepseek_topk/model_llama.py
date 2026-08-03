"""Exp 1 — DeepSeek Top-K sparse attention on R1-Distill-Llama-8B.

Selects the top-k most relevant keys per head (shared across query positions)
using a low-rank proxy, then attends only over those keys. This is the Llama-3
port of the original BART-based experiment.

Key changes from the BART version:
  - Inherits from ``LlamaSparseAttention`` (handles GQA, RoPE, projections)
  - ``sparse_attention()`` receives already-projected, RoPE'd, GQA-expanded
    Q/K/V as [BH, T, d] — same interface the sparse_attn_utils expect
  - Bidirectional attention (no causal mask) for sequence classification
  - LoRA training instead of full fine-tuning
"""

import os
import torch
import torch.nn as nn

from patches.llama.llama_patched_model import (
    LlamaSparseAttention,
    patch_llama,
    LlamaPatchedModel,
    apply_lora,
)
from sparse_attn_utils import (
    dense_self_attention,
    effective_top_k,
    head_shared_topk_indices,
    last_query_topk_indices,
    causal_sparse_attention,
    sdpa_head_shared_or_none,
    sparse_attention_head_shared,
)


class DeepSeekTopKAttention(LlamaSparseAttention):
    def __init__(self, base_attn, top_k: int = 128, low_rank_dim: int = 16, use_triton: bool = True):
        super().__init__(base_attn)
        self.top_k = top_k
        self.low_rank_dim = low_rank_dim
        self.use_triton = use_triton

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads, is_causal=False):
        BH, tgt_len, _ = Q.shape
        src_len = K.size(1)

        # Causal mode: use last-query routing + local window (O(N) attention)
        if is_causal:
            k_eff = effective_top_k(self.top_k, src_len, min_k=64, ratio=2)
            if src_len <= k_eff + 256:
                return dense_self_attention(
                    Q, K, V, token_mask, bsz, num_heads, 0.0, self.training,
                    is_causal=True,
                )
            d_low = min(self.low_rank_dim, self.head_dim)
            Q_low = Q[:, :, :d_low]
            K_low = K[:, :, :d_low]
            routed_idx = last_query_topk_indices(
                Q_low, K_low, k_eff, token_mask, bsz, num_heads,
            )
            return causal_sparse_attention(
                Q, K, V, routed_idx, local_window=256,
                token_mask=token_mask, bsz=bsz, num_heads=num_heads,
            )

        # Bidirectional mode: original head-shared routing
        k_eff = effective_top_k(self.top_k, src_len, min_k=64, ratio=2)

        if src_len <= k_eff:
            # Fall back to dense attention on short sequences
            return dense_self_attention(
                Q, K, V, token_mask, bsz, num_heads, 0.0, self.training,
                is_causal=is_causal,
            )

        d_low = min(self.low_rank_dim, self.head_dim)
        Q_low = Q[:, :, :d_low]
        K_low = K[:, :, :d_low]
        topk_idx = head_shared_topk_indices(
            Q_low, K_low, k_eff, token_mask, bsz, num_heads
        )
        out = sdpa_head_shared_or_none(
            Q, K, V, topk_idx, None, bsz, num_heads,
            self.use_triton, self.training, is_causal=is_causal,
        )
        if out is None:
            out = sparse_attention_head_shared(
                Q, K, V, topk_idx, 0.0, self.training, token_mask, bsz, num_heads,
                is_causal=is_causal,
            )
        return out


def build_model(
    model_path: str = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B"),
    top_k: int = 128,
    low_rank_dim: int = 64,
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    pooling: str = "last",
):
    """Build the patched R1-8B model with DeepSeek top-k attention + LoRA."""
    model = LlamaPatchedModel.from_pretrained(
        model_path=model_path,
        attention_cls=DeepSeekTopKAttention,
        num_labels=num_labels,
        attn_kwargs={
            "top_k": top_k,
            "low_rank_dim": low_rank_dim,
            "use_triton": False,  # safer on MIG; PyTorch fallback works
        },
        pooling=pooling,
    )
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha)
    return model
