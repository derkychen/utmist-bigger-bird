"""Exp 13 — Dynamic Context Window on R1-Distill-Llama-8B.

Caps the attention key set to a fixed ``target_budget`` tokens at every layer.
Importance is estimated from the L2 norm of the key vectors (a proxy for the
hidden-state norm used in the original BART version).

This is the Llama-3 port of the original BART-based experiment.  Key changes:
  - Inherits from ``LlamaSparseAttention`` (handles GQA, RoPE, projections)
  - ``sparse_attention()`` receives already-projected, RoPE'd, GQA-expanded
    Q/K/V as [BH, T, d] -- Q is pre-scaled, so no extra scaling is applied
  - ``self.layer_idx`` determines whether this layer is before or after
    ``drop_after_layer``
  - Token budgeting is implemented as attention masking (selecting a subset of
    keys) rather than physically removing tokens from the hidden-state sequence,
    since the HF LlamaModel controls the layer loop
  - The chunked early-layer path for very long sequences (PATH B in the
    original) is not applicable here because chunking requires inter-layer
    control; every layer instead uses bounded sparse routing
  - Bidirectional attention (no causal mask) for sequence classification
  - LoRA training instead of full fine-tuning
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
    sdpa_head_shared_or_none,
    sparse_attention_head_shared,
    causal_sparse_attention,
    last_query_topk_indices,
    head_shared_topk_indices,
)


class DynamicContextAttention(LlamaSparseAttention):
    """Budget-capped sparse attention at every layer.

    Each layer attends only over the top ``target_budget`` tokens ranked by
    key-vector L2 norm (averaged across heads within each batch element).
    """

    def __init__(
        self,
        base_attn,
        drop_after_layer: int = 3,
        target_budget: int = 4096,
        chunk_size: int = 8192,
        use_triton: bool = False,
    ):
        super().__init__(base_attn)
        self.drop_after_layer = drop_after_layer
        self.target_budget = target_budget
        self.chunk_size = chunk_size
        self.use_triton = use_triton

    def sparse_attention(self, Q, K, V, token_mask, bsz, num_heads, is_causal=False):
        BH, tgt_len, d = Q.shape
        src_len = K.size(1)

        # Causal mode: content-based top-k routing + local window (O(N) attention)
        # Uses last-query QK product (parameter-free) instead of key norm, because
        # norm-based selection is random for NIAH (the needle doesn't have a
        # distinctive key norm). Content-based routing finds the needle by its
        # distinctive QK similarity, matching exp 1's approach.
        if is_causal:
            k_eff = min(self.target_budget, src_len)
            d_low = min(d, 128)  # low-rank proxy for efficiency
            Q_low = Q[:, :, :d_low]
            K_low = K[:, :, :d_low]
            routed_idx = last_query_topk_indices(
                Q_low, K_low, k_eff, token_mask, bsz, num_heads,
            )
            return causal_sparse_attention(
                Q, K, V, routed_idx, local_window=256,
                token_mask=token_mask, bsz=bsz, num_heads=num_heads,
            )

        # --- Bidirectional: content-based top-k selection ---
        budget = min(self.target_budget, src_len)

        # Content-based top-k selection using head-shared QK product
        d_low = min(d, 128)
        Q_low = Q[:, :, :d_low]
        K_low = K[:, :, :d_low]
        indices = head_shared_topk_indices(
            Q_low, K_low, budget, token_mask, bsz, num_heads,
        )

        out = sdpa_head_shared_or_none(
            Q, K, V, indices, None, bsz, num_heads,
            self.use_triton, self.training, is_causal=is_causal,
        )
        if out is None:
            out = sparse_attention_head_shared(
                Q, K, V, indices, 0.0, self.training, token_mask, bsz, num_heads,
                is_causal=is_causal,
            )
        return out


def build_model(
    model_path: str = os.path.join(os.environ.get("SCRATCH", "/scratch/$USER"), "models", "DeepSeek-R1-Distill-Llama-8B"),
    drop_after_layer: int = 3,
    target_budget: int = 4096,
    chunk_size: int = 8192,
    num_labels: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    pooling: str = "last",
):
    """Build the patched R1-8B model with dynamic-context attention + LoRA."""
    model = LlamaPatchedModel.from_pretrained(
        model_path=model_path,
        attention_cls=DynamicContextAttention,
        num_labels=num_labels,
        attn_kwargs={
            "drop_after_layer": drop_after_layer,
            "target_budget": target_budget,
            "chunk_size": chunk_size,
            "use_triton": False,
        },
        pooling=pooling,
    )
    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha)
    return model
