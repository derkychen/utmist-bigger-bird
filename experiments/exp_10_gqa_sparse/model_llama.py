"""Exp 10 — GQA + Sparse Top-K attention on R1-Distill-Llama-8B.

Groups query heads into KV groups and applies top-k selection within each group.
This is the Llama-3 port of the original BART-based experiment.

Key changes from the BART version:
  - Llama-3 ALREADY has native GQA (32 query heads, 8 KV heads, 4:1 grouping).
    The base class (LlamaSparseAttention) expands KV heads to match query heads
    via repeat_interleave, so the GQA grouping is already done.
  - The sparse_attention() method just applies top-k selection on the
    already-projected, RoPE'd, GQA-expanded [BH, T, d] tensors.
  - The kv_groups parameter from BART maps to Llama's native num_kv_groups (4);
    no separate kv_groups parameter is needed.
  - Bidirectional attention (no causal mask) for sequence classification.
  - LoRA training instead of full fine-tuning.
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
    sdpa_head_shared_or_none,
    sparse_attention_head_shared,
)


class GQASparseLlamaAttention(LlamaSparseAttention):
    """GQA + sparse top-k attention for Llama-3.

    Since Llama-3 already has native GQA (KV heads are expanded by the base
    class), this module simply applies head-shared top-k key selection using
    a low-rank proxy, then attends over the selected keys.
    """

    def __init__(self, base_attn, top_k: int = 64, low_rank_dim: int = 16, use_triton: bool = True):
        super().__init__(base_attn)
        self.top_k = top_k
        self.low_rank_dim = low_rank_dim
        self.use_triton = use_triton

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads):
        BH, tgt_len, _ = Q.shape
        src_len = K.size(1)
        k_eff = effective_top_k(self.top_k, src_len)

        if src_len <= k_eff:
            # Fall back to dense attention on short sequences
            return dense_self_attention(
                Q, K, V, token_mask, bsz, num_heads, 0.0, self.training
            )

        # Low-rank proxy for top-k selection.
        # In the BART version, K was compressed within each GQA group before
        # top-k selection. Since Llama-3's KV heads are already shared within
        # each group (expanded by repeat_interleave in the base class), all
        # heads in a group have identical K — so using K directly is equivalent
        # to the GQA-compressed version.
        d_low = min(self.low_rank_dim, self.head_dim)
        Q_low = Q[:, :, :d_low]
        K_low = K[:, :, :d_low]
        topk_idx = head_shared_topk_indices(
            Q_low, K_low, k_eff, token_mask, bsz, num_heads
        )
        out = sdpa_head_shared_or_none(
            Q, K, V, topk_idx, None, bsz, num_heads,
            self.use_triton, self.training,
        )
        if out is None:
            out = sparse_attention_head_shared(
                Q, K, V, topk_idx, 0.0, self.training, token_mask, bsz, num_heads
            )
        return out


def build_model(
    model_path: str = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B"),
    top_k: int = 64,
    low_rank_dim: int = 16,
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """Build the patched R1-8B model with GQA sparse top-k attention + LoRA."""
    model = LlamaPatchedModel.from_pretrained(
        model_path=model_path,
        attention_cls=GQASparseLlamaAttention,
        num_labels=num_labels,
        attn_kwargs={
            "top_k": top_k,
            "low_rank_dim": low_rank_dim,
            "use_triton": False,  # safer on MIG; PyTorch fallback works
        },
    )
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha)
    return model
