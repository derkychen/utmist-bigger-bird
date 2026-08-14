"""Exp 7 — Layer-Adaptive Top-K Attention on R1-Distill-Llama-8B.

Uses different top-k values for early, mid, and late layers. Early layers
attend more broadly (k_early=192), while later layers use progressively
sparser attention (k_mid=64, k_late=32). The layer index is read from
``self.layer_idx`` (set by the base class from the HF model).

Llama-8B has 32 layers. With the thirds-based schedule:
  - Layers 0-10:  early (k=192)
  - Layers 11-21: mid   (k=64)
  - Layers 22-31: late  (k=32)

Key changes from the BART version:
  - Inherits from ``LlamaSparseAttention`` (handles GQA, RoPE, projections)
  - ``sparse_attention()`` receives already-projected, RoPE'd, GQA-expanded
    Q/K/V as [BH, T, d] — Q is pre-scaled by self.scaling
  - Bidirectional attention (no causal mask) for sequence classification
  - LoRA training instead of full fine-tuning
  - k is selected inside __init__ based on self.layer_idx and
    self.config.num_hidden_layers (32 for Llama-8B)
"""

import os
import torch
import torch.nn as nn

from patches.llama.llama_patched_model import (
    LlamaSparseAttention,
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


def _schedule(layer_idx: int, n_layers: int, k_early: int, k_mid: int, k_late: int):
    third = n_layers / 3
    if layer_idx < third:
        return k_early
    if layer_idx < 2 * third:
        return k_mid
    return k_late


class LayerAdaptiveAttention(LlamaSparseAttention):
    def __init__(
        self,
        base_attn,
        k_early: int = 192,
        k_mid: int = 64,
        k_late: int = 32,
        low_rank_dim: int = 16,
        use_triton: bool = True,
    ):
        super().__init__(base_attn)
        self.k_early = k_early
        self.k_mid = k_mid
        self.k_late = k_late
        self.low_rank_dim = low_rank_dim
        self.use_triton = use_triton
        # Pick k based on this layer's position in the model
        n_layers = self.config.num_hidden_layers  # 32 for Llama-8B
        self.top_k = _schedule(
            self.layer_idx, n_layers, k_early, k_mid, k_late
        )

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads, is_causal=False):
        BH, tgt_len, _ = Q.shape
        src_len = K.size(1)
        k_eff = effective_top_k(self.top_k, src_len)

        # Causal mode: last-query routing + local window (O(N) attention)
        if is_causal:
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

        if src_len <= k_eff:
            # Fall back to dense attention on short sequences
            return dense_self_attention(
                Q, K, V, None, bsz, num_heads, 0.0, self.training
            )

        # --- Low-rank head-shared top-k selection ---
        d_low = min(self.low_rank_dim, self.head_dim)
        Q_low = Q[:, :, :d_low]
        K_low = K[:, :, :d_low]
        topk_idx = head_shared_topk_indices(
            Q_low, K_low, k_eff, token_mask, bsz, num_heads
        )

        # --- Sparse attention over selected keys ---
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
    k_early: int = 192,
    k_mid: int = 64,
    k_late: int = 32,
    low_rank_dim: int = 16,
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """Build the patched R1-8B model with layer-adaptive top-k attention + LoRA."""
    model = LlamaPatchedModel.from_pretrained(
        model_path=model_path,
        attention_cls=LayerAdaptiveAttention,
        num_labels=num_labels,
        attn_kwargs={
            "k_early": k_early,
            "k_mid": k_mid,
            "k_late": k_late,
            "low_rank_dim": low_rank_dim,
            "use_triton": False,
        },
    )
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha)
    return model
