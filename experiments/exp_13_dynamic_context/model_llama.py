"""Exp 13 — Dynamic Context Window on R1-Distill-Llama-8B.

Caps the attention key set to a fixed ``target_budget`` tokens after a few
early dense layers.  Importance is estimated from the L2 norm of the key
vectors (a proxy for the hidden-state norm used in the original BART version).
Early layers use full dense attention so that local syntax is extracted before
the budget is applied.

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
    control; early layers simply use dense attention
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
    dense_self_attention,
    sdpa_head_shared_or_none,
    sparse_attention_head_shared,
    causal_sparse_attention,
)


class DynamicContextAttention(LlamaSparseAttention):
    """Dense attention for early layers; budget-capped sparse attention later.

    Layers with ``layer_idx < drop_after_layer`` use full dense attention.
    Later layers attend only over the top ``target_budget`` tokens ranked by
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

        # Causal mode: budget-capped norm selection + local window (O(N) attention)
        if is_causal:
            budget = min(self.target_budget, src_len)
            if src_len <= budget + 256:
                return dense_self_attention(
                    Q, K, V, token_mask, bsz, num_heads, 0.0, self.training,
                    is_causal=True,
                )
            K_reshaped = K.view(bsz, num_heads, src_len, d)
            norms = K_reshaped.norm(dim=-1).mean(dim=1)  # [B, T]
            if token_mask is not None:
                norms = norms.masked_fill(~token_mask, -1e9)
            _, top_idx = torch.topk(norms, k=budget, dim=-1)
            top_idx, _ = torch.sort(top_idx, dim=-1)
            indices = top_idx.unsqueeze(1).expand(bsz, num_heads, budget).reshape(BH, budget)
            return causal_sparse_attention(
                Q, K, V, indices, local_window=256,
                token_mask=token_mask, bsz=bsz, num_heads=num_heads,
            )

        # --- Early layers: full dense attention ---
        if self.layer_idx < self.drop_after_layer:
            return dense_self_attention(
                Q, K, V, token_mask, bsz, num_heads, 0.0, self.training,
                is_causal=is_causal,
            )

        # --- Later layers: attend over top target_budget tokens ---
        budget = min(self.target_budget, src_len)
        if budget >= src_len:
            return dense_self_attention(
                Q, K, V, token_mask, bsz, num_heads, 0.0, self.training,
                is_causal=is_causal,
            )

        # Importance proxy: key-vector L2 norm averaged across heads -> [B, T]
        K_reshaped = K.view(bsz, num_heads, src_len, d)
        norms = K_reshaped.norm(dim=-1).mean(dim=1)  # [B, T]
        if token_mask is not None:
            norms = norms.masked_fill(~token_mask, -1e9)

        _, top_idx = torch.topk(norms, k=budget, dim=-1)  # [B, budget]
        top_idx, _ = torch.sort(top_idx, dim=-1)  # preserve relative order

        # Expand to [BH, budget] -- same indices for all heads within a batch
        indices = top_idx.unsqueeze(1).expand(bsz, num_heads, budget).reshape(BH, budget)

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
